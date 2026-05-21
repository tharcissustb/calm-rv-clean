from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "liar_dataset"
OUT_PATH = ROOT / "data" / "liar.jsonl"

LABEL_MAP = {
    "true": "true",
    "mostly-true": "true",
    "false": "false",
    "pants-fire": "false",
    "half-true": "unverified",
    "barely-true": "unverified",
}

# LIAR format:
# 0=id, 1=label, 2=statement, ...
ID_COL = 0
LABEL_COL = 1
TEXT_COL = 2


def read_tsv(path: Path, split_name: str):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, cols in enumerate(reader):
            if len(cols) <= TEXT_COL:
                continue

            raw_label = cols[LABEL_COL].strip().lower()
            if raw_label not in LABEL_MAP:
                continue

            text = cols[TEXT_COL].strip()
            if not text:
                continue

            ex_id = cols[ID_COL].strip() or f"{split_name}_{i}"

            rows.append(
                {
                    "id": f"liar_{split_name}_{ex_id}",
                    "dataset": "liar",
                    "event_id": split_name,
                    "thread_id": ex_id,
                    "text": text,
                    "label": LABEL_MAP[raw_label],
                    "original_label": raw_label,
                    "source_split": split_name,
                }
            )
    return rows


def main():
    all_rows = []
    stats = {}

    for split_name, fname in [
        ("train", "train.tsv"),
        ("valid", "valid.tsv"),
        ("test", "test.tsv"),
    ]:
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

        rows = read_tsv(path, split_name)
        all_rows.extend(rows)
        stats[split_name] = len(rows)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 50)
    print("LIAR JSONL created:", OUT_PATH)
    print("Counts by split:", stats)
    print("Total examples:", len(all_rows))
    print("=" * 50)


if __name__ == "__main__":
    main()