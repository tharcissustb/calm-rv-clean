from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "RumourEval17" / "semeval2017-task8-dataset"
DATA_DIR = BASE_DIR / "rumoureval-data"
TRAINDEV_DIR = BASE_DIR / "traindev"
OUT_PATH = ROOT / "data" / "rumoureval17.jsonl"

VALID_LABELS = {"true", "false", "unverified"}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_source_text(thread_dir: Path) -> str:
    candidates = list((thread_dir / "source-tweet").glob("*.json")) + list((thread_dir / "source-tweets").glob("*.json"))
    if not candidates:
        return ""
    try:
        obj = load_json(candidates[0])
        if isinstance(obj, dict):
            return str(obj.get("text", "")).strip()
    except Exception:
        return ""
    return ""


def main():
    train_labels_path = TRAINDEV_DIR / "rumoureval-subtaskB-train.json"
    dev_labels_path = TRAINDEV_DIR / "rumoureval-subtaskB-dev.json"

    if not train_labels_path.exists():
        raise FileNotFoundError(f"Missing: {train_labels_path}")
    if not dev_labels_path.exists():
        raise FileNotFoundError(f"Missing: {dev_labels_path}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing: {DATA_DIR}")

    train_labels = load_json(train_labels_path)
    dev_labels = load_json(dev_labels_path)

    labels = {}
    labels.update(train_labels)
    labels.update(dev_labels)

    rows = []
    skipped = {
        "missing_thread_dir": 0,
        "missing_text": 0,
        "bad_label": 0,
    }

    # Build index of thread folder name -> full path
    thread_dir_index = {}
    for event_dir in DATA_DIR.iterdir():
        if not event_dir.is_dir():
            continue
        for thread_dir in event_dir.iterdir():
            if thread_dir.is_dir():
                thread_dir_index[thread_dir.name] = (event_dir.name, thread_dir)

    for thread_id, label in labels.items():
        label = str(label).strip().lower()
        if label not in VALID_LABELS:
            skipped["bad_label"] += 1
            continue

        if thread_id not in thread_dir_index:
            skipped["missing_thread_dir"] += 1
            continue

        event_id, thread_dir = thread_dir_index[thread_id]
        text = read_source_text(thread_dir)
        if not text:
            skipped["missing_text"] += 1
            continue

        rows.append(
            {
                "id": f"rumoureval17_{event_id}_{thread_id}",
                "dataset": "rumoureval17",
                "event_id": event_id,
                "thread_id": thread_id,
                "text": text,
                "label": label,
            }
        )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 50)
    print("RumourEval17 JSONL created:", OUT_PATH)
    print("Total labels:", len(labels))
    print("Total examples written:", len(rows))
    print("Skipped:", skipped)
    print("=" * 50)


if __name__ == "__main__":
    main()