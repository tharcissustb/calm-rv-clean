"""
Adaptive Calibration for Cross-Event Generalization
Based on run_loeo.py with added adaptive calibration using event difficulty scores
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.evaluator import evaluate_and_save

LABEL_MAP = {"true": 0, "false": 1, "unverified": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Pre-computed difficulty scores from your analysis
EVENT_DIFFICULTY = {
    "charliehebdo": 0.0019,
    "ebola-essien": 0.0018,
    "ferguson": 0.0025,
    "germanwings-crash": 0.0020,
    "gurlitt": 0.0025,
    "ottawashooting": 0.0020,
    "prince-toronto": 0.0021,
    "putinmissing": 0.0020,
    "sydneysiege": 0.0018,
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(force_cpu: bool):
    if (not force_cpu) and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    return torch.device("cpu")


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if "text" in row and "label" in row and "event_id" in row:
                    rows.append(row)
            except Exception:
                continue
    return rows


def split_train_val(rows, seed=42, val_ratio=0.1):
    rng = random.Random(seed)
    by_label = {}

    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    train, val = [], []

    for _, items in by_label.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class RumourDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(LABEL_MAP[row["label"]], dtype=torch.long)
        item["id"] = row.get("id", str(idx))
        item["event_id"] = row.get("event_id", "unknown")
        return item


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, y_true, logits_all = [], [], []

    for batch in loader:
        batch_ids = batch.pop("id")
        batch_events = batch.pop("event_id")
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits.detach().float().cpu().numpy()

        ids.extend(batch_ids)
        y_true.extend(labels.numpy())
        logits_all.append(logits)

    return ids, y_true, np.concatenate(logits_all, axis=0)


def train_one_fold(train_rows, val_rows, test_rows, cfg, output_dir, device, use_adaptive=True):
    """
    Train with optional adaptive calibration based on event difficulty
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
    )
    model.to(device)

    train_ds = RumourDataset(train_rows, tokenizer, cfg["model"]["max_length"])
    val_ds = RumourDataset(val_rows, tokenizer, cfg["model"]["max_length"])
    test_ds = RumourDataset(test_rows, tokenizer, cfg["model"]["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))

    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Calibration hyperparameters
    base_lambda_cal = cfg.get("calibration", {}).get("lambda_cal", 0.5)
    adaptive_scale = cfg.get("calibration", {}).get("adaptive_scale", 10.0)
    difficulty_threshold = cfg.get("calibration", {}).get("difficulty_threshold", 0.002)

    best_f1 = -1.0
    best_state = None

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Get event_id for adaptive calibration
            batch_events = batch.pop("event_id")
            batch.pop("id")
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch, labels=labels)
            loss = outputs.loss

            # ADAPTIVE CALIBRATION LOSS
            if use_adaptive:
                # Get difficulty for each sample in batch
                batch_difficulties = []
                for event in batch_events:
                    diff = EVENT_DIFFICULTY.get(event, 0.002)
                    batch_difficulties.append(diff)
                
                avg_difficulty = np.mean(batch_difficulties)
                
                # Adaptive lambda: higher difficulty = higher calibration weight
                if avg_difficulty > difficulty_threshold:
                    adaptive_lambda = base_lambda_cal * (1 + adaptive_scale * (avg_difficulty - difficulty_threshold))
                else:
                    adaptive_lambda = base_lambda_cal
                
                # Compute calibration loss (Brier score)
                probs = torch.softmax(outputs.logits, dim=-1)
                one_hot = torch.nn.functional.one_hot(labels, num_classes=3).float()
                cal_loss = torch.mean((probs - one_hot) ** 2)
                
                loss = loss + adaptive_lambda * cal_loss
                
                # Log for debugging
                if epoch == 0 and len(batch_events) > 0:
                    print(f"    Event: {batch_events[0]}, Difficulty: {avg_difficulty:.4f}, λ_cal: {adaptive_lambda:.2f}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        val_ids, val_y, val_logits = predict(model, val_loader, device)
        val_metrics = evaluate_and_save(
            output_dir,
            f"val_epoch{epoch+1}",
            val_ids,
            val_y,
            val_logits,
            15,
            ID_TO_LABEL,
        )

        print(f"Epoch {epoch+1} | loss={avg_loss:.4f} | val_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_ids, test_y, test_logits = predict(model, test_loader, device)
    test_metrics = evaluate_and_save(
        output_dir,
        "test",
        test_ids,
        test_y,
        test_logits,
        15,
        ID_TO_LABEL,
    )

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--min_event_size", type=int, default=0)
    parser.add_argument("--no_adaptive", action="store_true", help="Disable adaptive calibration (use fixed)")
    parser.add_argument("--hard_events_only", action="store_true", help="Only run on hard events (difficulty > 0.002)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    set_seed(cfg["training"]["seed"])
    device = choose_device(args.force_cpu)
    print("Using device:", device)
    print(f"Adaptive calibration: {not args.no_adaptive}")
    
    if not args.no_adaptive:
        print(f"  Base λ_cal: {cfg.get('calibration', {}).get('lambda_cal', 0.5)}")
        print(f"  Adaptive scale: {cfg.get('calibration', {}).get('adaptive_scale', 10.0)}")
        print(f"  Difficulty threshold: {cfg.get('calibration', {}).get('difficulty_threshold', 0.002)}")

    rows = load_jsonl(ROOT / cfg["dataset"]["path"])

    by_event = {}
    for r in rows:
        by_event.setdefault(r["event_id"], []).append(r)

    events = sorted(by_event.keys())
    
    # Filter to hard events if requested
    if args.hard_events_only:
        hard_events = [e for e in events if EVENT_DIFFICULTY.get(e, 0) >= 0.002]
        print(f"\nFiltering to hard events only: {hard_events}")
        events = hard_events

    run_dir = ROOT / "outputs" / "runs" / f"{args.exp_name}_seed{cfg['training']['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for event in events:
        test_rows = by_event[event]
        train_pool = []

        for ev, ev_rows in by_event.items():
            if ev != event:
                train_pool.extend(ev_rows)

        train_rows, val_rows = split_train_val(
            train_pool,
            seed=cfg["training"]["seed"],
            val_ratio=0.1,
        )

        fold_dir = run_dir / f"loeo_test_{event}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"[LOEO] test_event={event}")
        print(f"  Difficulty: {EVENT_DIFFICULTY.get(event, 0.002):.4f}")
        print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")
        print(f"{'='*50}")

        metrics = train_one_fold(
            train_rows, val_rows, test_rows, cfg, fold_dir, device,
            use_adaptive=not args.no_adaptive
        )
        metrics["n_test"] = len(test_rows)
        metrics["difficulty"] = EVENT_DIFFICULTY.get(event, 0.002)

        all_results[event] = metrics
        print(f"TEST Macro-F1: {metrics['macro_f1']:.4f}, ECE: {metrics['ece']:.4f}")

    # Save results
    with open(run_dir / "loeo_results_adaptive.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    macro_f1s = [v["macro_f1"] for v in all_results.values()]
    ece_scores = [v["ece"] for v in all_results.values()]
    difficulties = [v["difficulty"] for v in all_results.values()]

    summary = {
        "exp_name": args.exp_name,
        "adaptive_enabled": not args.no_adaptive,
        "hard_events_only": args.hard_events_only,
        "num_events": len(all_results),
        "avg_macro_f1": float(np.mean(macro_f1s)),
        "avg_ece": float(np.mean(ece_scores)),
        "worst_macro_f1": float(np.min(macro_f1s)),
        "worst_event": list(all_results.keys())[np.argmin(macro_f1s)],
        "per_event_results": all_results,
    }

    with open(run_dir / "loeo_summary_adaptive.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Adaptive: {not args.no_adaptive}")
    print(f"Events tested: {len(all_results)}")
    print(f"Average Macro-F1: {summary['avg_macro_f1']:.4f}")
    print(f"Average ECE: {summary['avg_ece']:.4f}")
    print(f"Worst Macro-F1: {summary['worst_macro_f1']:.4f} ({summary['worst_event']})")
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()