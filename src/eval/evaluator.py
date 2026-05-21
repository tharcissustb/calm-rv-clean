from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .metrics import classification_metrics, softmax


def save_predictions_jsonl(
    out_path: Path,
    ids: list[str],
    labels: list[int],
    logits: np.ndarray,
    label_id_to_name: dict[int, str] | None = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    probs = softmax(logits, axis=1)
    preds = probs.argmax(axis=1)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(len(ids)):
            rec = {
                "id": ids[i],
                "y_true": int(labels[i]),
                "y_pred": int(preds[i]),
                "probs": [float(x) for x in probs[i].tolist()],
            }
            if label_id_to_name is not None:
                rec["y_true_name"] = label_id_to_name[int(labels[i])]
                rec["y_pred_name"] = label_id_to_name[int(preds[i])]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def evaluate_and_save(
    run_dir: Path,
    split_name: str,
    ids: list[str],
    y_true: list[int],
    logits: np.ndarray,
    ece_bins: int = 15,
    label_id_to_name: dict[int, str] | None = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = classification_metrics(logits=logits, y_true=np.array(y_true), n_bins=ece_bins)

    # Save metrics
    metrics_path = run_dir / f"metrics_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    preds_path = run_dir / f"predictions_{split_name}.jsonl"
    save_predictions_jsonl(preds_path, ids, y_true, logits, label_id_to_name=label_id_to_name)

    return metrics