"""
CALM-RV LOEO Training Framework with Multi-Seed Support
Includes: L_clean, L_align, L_cal, L_robust + LLM augmentation
Supports --seed argument for statistical validation
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.contrastive_loss import ContrastiveAlignmentLoss
from src.eval.metrics import classification_metrics

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        item["text"] = row["text"]
        return item


def get_representations(model, input_ids, attention_mask):
    if hasattr(model, 'roberta'):
        outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    elif hasattr(model, 'bert'):
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    return outputs.last_hidden_state[:, 0, :]


def brier_loss(logits, labels, num_classes=3):
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(labels, num_classes).float()
    return torch.mean((probs - one_hot) ** 2)


def simple_perturb(text, p=0.05):
    words = text.split()
    if len(words) < 3:
        return text
    if random.random() < p:
        idx = random.randint(0, len(words) - 1)
        words[idx] = words[idx].lower()
    return " ".join(words)


def train_one_fold(train_rows, val_rows, test_rows, test_event_name, args, cfg, device):
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

    base_lambda_cal = cfg.get("calibration", {}).get("lambda_cal", 0.5)

    best_f1 = -1.0
    best_state = None
    
    contrastive_loss_fn = ContrastiveAlignmentLoss(temperature=0.07)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            texts = batch.get("text", [])
            batch_events = batch.get("event_id", [])
            batch.pop("id", None)
            batch.pop("text", None)
            batch.pop("event_id", None)
            labels = batch.pop("labels").to(device)
            
            input_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
            
            optimizer.zero_grad()
            outputs = model(**input_batch, labels=labels)
            loss = outputs.loss

            reps = get_representations(model, input_batch["input_ids"], input_batch["attention_mask"])

            if args.use_align:
                align_loss = contrastive_loss_fn(reps, labels, batch_events)
                loss = loss + args.lambda_align * align_loss

            if args.use_calibration:
                cal_loss = brier_loss(outputs.logits, labels)
                loss = loss + base_lambda_cal * cal_loss

            if args.use_robust:
                perturbed = [simple_perturb(t) for t in texts]
                pert_enc = tokenizer(perturbed, padding=True, truncation=True, max_length=128, return_tensors="pt")
                pert_input = {k: v.to(device) for k, v in pert_enc.items() if hasattr(v, 'to')}
                pert_outputs = model(**pert_input)
                robust_loss = F.cross_entropy(pert_outputs.logits, labels)
                loss = loss + args.lambda_robust * robust_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        model.eval()
        all_labels, all_logits = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch.pop("id", None)
                batch.pop("text", None)
                batch.pop("event_id", None)
                labels = batch.pop("labels").numpy()
                input_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
                outputs = model(**input_batch)
                logits = outputs.logits.cpu().numpy()
                all_labels.extend(labels)
                all_logits.append(logits)

        all_logits = np.vstack(all_logits)
        val_metrics = classification_metrics(all_logits, np.array(all_labels))

        print(f"Epoch {epoch+1} | loss={avg_loss:.4f} | val_f1={val_metrics['macro_f1']:.4f} | val_ece={val_metrics['ece']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_labels, all_logits = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch.pop("id", None)
            batch.pop("text", None)
            batch.pop("event_id", None)
            labels = batch.pop("labels").numpy()
            input_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
            outputs = model(**input_batch)
            logits = outputs.logits.cpu().numpy()
            all_labels.extend(labels)
            all_logits.append(logits)

    all_logits = np.vstack(all_logits)
    test_metrics = classification_metrics(all_logits, np.array(all_labels))

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    parser.add_argument("--test_event", default="ferguson", choices=[
        'charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash',
        'gurlitt', 'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege'
    ])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--use_align", action="store_true", help="Use contrastive alignment loss")
    parser.add_argument("--use_calibration", action="store_true", help="Use calibration loss")
    parser.add_argument("--use_robust", action="store_true", help="Use robustness loss")
    parser.add_argument("--use_augmented", action="store_true", help="Use augmented data")
    parser.add_argument("--use_adaptive", action="store_true", help="Use adaptive calibration")
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_cal", type=float, default=0.5)
    parser.add_argument("--lambda_robust", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.force_cpu)
    
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cfg["training"]["seed"] = args.seed

    print("="*60)
    print(f"LOEO Experiment: {args.exp_name}")
    print(f"Test event: {args.test_event}")
    print(f"  Align: {args.use_align} (λ={args.lambda_align})")
    print(f"  Calibration: {args.use_calibration} (λ={args.lambda_cal})")
    print(f"  Robust: {args.use_robust} (λ={args.lambda_robust})")
    print(f"  Augmented: {args.use_augmented}")
    print(f"  Adaptive: {args.use_adaptive}")
    print("="*60)

    # Load data
    data_path = ROOT / cfg["dataset"]["path"]
    all_rows = load_jsonl(data_path)

    by_event = {}
    for r in all_rows:
        by_event.setdefault(r["event_id"], []).append(r)

    test_rows = by_event.get(args.test_event, [])
    train_pool = []
    for ev, ev_rows in by_event.items():
        if ev != args.test_event:
            train_pool.extend(ev_rows)

    train_rows, val_rows = split_train_val(train_pool, seed=args.seed, val_ratio=0.1)

    if args.use_augmented:
        aug_path = ROOT / "data/augmented/hard_augmented.jsonl"
        if aug_path.exists():
            augmented_rows = load_jsonl(aug_path)
            train_rows = train_rows + augmented_rows
            print(f"  Added {len(augmented_rows)} augmented examples")

    print(f"Train size: {len(train_rows)}")
    print(f"Test size: {len(test_rows)}")

    out_dir = ROOT / "outputs/loeo" / f"{args.exp_name}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = train_one_fold(train_rows, val_rows, test_rows, args.test_event, args, cfg, device)

    with open(out_dir / f"{args.test_event}_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*60)
    print(f"RESULTS FOR {args.test_event.upper()} (seed={args.seed})")
    print("="*60)
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"Brier: {metrics['brier']:.4f}")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()