from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.evaluator import evaluate_and_save

LABEL_MAP = {"true": 0, "false": 1, "unverified": 2}


def choose_device(force_cpu=True):
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
            except Exception:
                continue
    return rows


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
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    source_run = ROOT / "outputs" / "runs" / args.source_run
    target_jsonl = ROOT / "data" / args.target
    out_dir = ROOT / "outputs" / "runs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=3,
    )

    device = choose_device(args.force_cpu)
    state = torch.load(source_run / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    rows = load_jsonl(target_jsonl)
    ds = RumourDataset(rows, tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    ids, y_true, logits_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch_ids = batch.pop("id")
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()

            ids.extend(batch_ids)
            y_true.extend(labels.numpy())
            logits_all.append(logits)

    logits_all = np.concatenate(logits_all, axis=0)

    metrics = evaluate_and_save(
        run_dir=out_dir,
        split_name="test",
        ids=ids,
        y_true=y_true,
        logits=logits_all,
        ece_bins=10,
        label_id_to_name={0: "true", 1: "false", 2: "unverified"},
    )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("DONE:", args.out)
    print("METRICS:", metrics)


if __name__ == "__main__":
    main()