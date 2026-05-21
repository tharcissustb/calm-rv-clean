from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def export_indomain(run_dir: Path, out_dir: Path):
    summary = load_json(run_dir / "summary.json")

    row = {
        "run_id": run_dir.name,
        "accuracy": summary.get("accuracy"),
        "macro_f1": summary.get("macro_f1"),
        "ece": summary.get("ece"),
        "brier": summary.get("brier"),
        "nll": summary.get("nll"),
    }

    df = pd.DataFrame([row])
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / f"{run_dir.name}_summary.csv", index=False)
    df.to_excel(out_dir / f"{run_dir.name}_summary.xlsx", index=False)

    print("INDOMAIN EXPORT DONE")

def export_loeo(run_dir: Path, out_dir: Path):
    results = load_json(run_dir / "loeo_results.json")

    rows = []
    for event, vals in results.items():
        rows.append({
            "event": event,
            "accuracy": vals.get("accuracy"),
            "macro_f1": vals.get("macro_f1"),
        })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / f"{run_dir.name}_loeo.csv", index=False)

    print("LOEO EXPORT DONE")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--mode", required=True, choices=["indomain", "loeo"])
    parser.add_argument("--out_dir", default="outputs/tables")
    args = parser.parse_args()

    run_dir = ROOT / args.run_dir
    out_dir = ROOT / args.out_dir

    print("RUN DIR:", run_dir)

    if args.mode == "indomain":
        export_indomain(run_dir, out_dir)
    else:
        export_loeo(run_dir, out_dir)

if __name__ == "__main__":
    main()
