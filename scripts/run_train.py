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

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.evaluator import evaluate_and_save


# =====================
# LABELS
# =====================
LABEL_MAP = {"true": 0, "false": 1, "unverified": 2}


# =====================
# UTILITIES
# =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(force_cpu: bool):
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if "text" in row and "label" in row:
                    rows.append(row)
            except:
                continue
    return rows


def stratified_split(rows, seed=42):
    rng = random.Random(seed)
    by_label = {}

    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    train, val, test = [], [], []

    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)

        train.extend(items[: int(0.7 * n)])
        val.extend(items[int(0.7 * n): int(0.8 * n)])
        test.extend(items[int(0.8 * n):])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# =====================
# PERTURBATION
# =====================
def simple_perturb(text):
    words = text.split()
    if len(words) > 5:
        i = random.randint(0, len(words) - 1)
        words[i] = words[i].lower()
    return " ".join(words)


# =====================
# DATASET
# =====================
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
        item["labels"] = torch.tensor(LABEL_MAP[row["label"]])
        item["id"] = row.get("id", str(idx))
        item["text"] = row["text"]
        return item


# =====================
# PREDICT (FIXED)
# =====================
@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    ids, y_true = [], []
    logits_list = []

    for batch in loader:
        batch_ids = batch.pop("id")
        batch.pop("text")
        labels = batch.pop("labels")

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits.detach().cpu().numpy()

        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        logits_list.append(logits)
        ids.extend(batch_ids)
        y_true.extend(labels.numpy())

    if len(logits_list) == 0:
        raise RuntimeError("No logits collected!")

    logits_all = np.vstack(logits_list)

    return ids, y_true, logits_all


# =====================
# MAIN
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    seed = cfg["training"]["seed"]
    set_seed(seed)

    # ABLATION FLAGS
    ablation = cfg.get("ablation", {})
    use_robust = ablation.get("use_robust", False)
    use_calibration = ablation.get("use_calibration", False)
    lambda_robust = ablation.get("lambda_robust", 1.0)
    lambda_cal = ablation.get("lambda_cal", 0.5)

    print("Ablation settings:", use_robust, use_calibration)

    # DATA
    rows = load_jsonl(ROOT / cfg["dataset"]["path"])
    train_rows, val_rows, test_rows = stratified_split(rows, seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
    )

    device = choose_device(args.force_cpu)
    model.to(device)

    train_loader = DataLoader(RumourDataset(train_rows, tokenizer, cfg["model"]["max_length"]), batch_size=16, shuffle=True)
    val_loader = DataLoader(RumourDataset(val_rows, tokenizer, cfg["model"]["max_length"]), batch_size=16)
    test_loader = DataLoader(RumourDataset(test_rows, tokenizer, cfg["model"]["max_length"]), batch_size=16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_f1 = -1
    best_state = None

    out_dir = ROOT / "outputs/runs" / f"{args.exp_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # =====================
    # TRAIN
    # =====================
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0

        for batch in train_loader:
            texts = batch.pop("text")
            batch.pop("id")
            labels = batch.pop("labels").to(device)

            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(**batch, labels=labels)
            loss = outputs.loss

            # ROBUST
            if use_robust:
                perturbed = [simple_perturb(t) for t in texts]
                enc = tokenizer(perturbed, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                loss += lambda_robust * model(**enc, labels=labels).loss

            # CALIBRATION
            if use_calibration:
                probs = torch.softmax(outputs.logits, dim=-1)
                one_hot = torch.nn.functional.one_hot(labels, num_classes=3).float()
                loss += lambda_cal * torch.mean((probs - one_hot) ** 2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        ids, y_true, logits = predict(model, val_loader, device)

        print("DEBUG logits:", logits.shape)

        results = evaluate_and_save(out_dir, ids, y_true, logits, f"val_epoch{epoch+1}")

        print(f"Epoch {epoch+1} | loss={total_loss:.4f} | val_f1={results['macro_f1']:.4f}")

        if results["macro_f1"] > best_f1:
            best_f1 = results["macro_f1"]
            best_state = model.state_dict()

    # SAVE
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pt")

    # TEST
    ids, y_true, logits = predict(model, test_loader, device)
    results = evaluate_and_save(out_dir, ids, y_true, logits, "test")

    print("FINAL TEST:", results)


if __name__ == "__main__":
    main()