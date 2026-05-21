import argparse, json
from pathlib import Path
from collections import OrderedDict

import numpy as np

import matplotlib.pyplot as plt

# Excel writer
try:
    import pandas as pd
except ImportError as e:
    raise ImportError("Please install pandas + openpyxl: pip install pandas openpyxl") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to LOEO run folder, e.g. outputs/runs/pheme_roberta_ce_loeo_seed42")
    ap.add_argument("--out_tables", default="outputs/tables")
    ap.add_argument("--out_figures", default="outputs/figures")
    ap.add_argument("--min_event_size", type=int, default=0, help="Filter events with n_test < min_event_size for the SUMMARY only")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    res_path = run_dir / "loeo_results.json"
    if not res_path.exists():
        raise FileNotFoundError(f"Missing: {res_path}")

    results = json.loads(res_path.read_text(encoding="utf-8"))

    # deterministic ordering
    events = sorted(results.keys())

    rows = []
    for e in events:
        r = results[e]
        rows.append({
            "event": e,
            "n_test": int(r.get("n_test", 0)),
            "accuracy": float(r.get("accuracy", float("nan"))),
            "macro_f1": float(r.get("macro_f1", float("nan"))),
            "ece": float(r.get("ece", float("nan"))),
            "brier": float(r.get("brier", float("nan"))),
            "nll": float(r.get("nll", float("nan"))),
        })

    df = pd.DataFrame(rows).sort_values("event").reset_index(drop=True)

    # summary (optionally filtered)
    df_used = df[df["n_test"] >= args.min_event_size].copy()
    avg_f1 = float(df_used["macro_f1"].mean()) if len(df_used) else float("nan")
    worst_idx = int(df_used["macro_f1"].idxmin()) if len(df_used) else None
    worst_event = df_used.loc[worst_idx, "event"] if worst_idx is not None else None
    worst_f1 = float(df_used.loc[worst_idx, "macro_f1"]) if worst_idx is not None else float("nan")

    summary = pd.DataFrame([{
        "run_dir": str(run_dir),
        "min_event_size": int(args.min_event_size),
        "num_events_total": int(len(df)),
        "num_events_used": int(len(df_used)),
        "avg_macro_f1": avg_f1,
        "worst_event": worst_event,
        "worst_event_macro_f1": worst_f1,
    }])

    out_tables = Path(args.out_tables); out_tables.mkdir(parents=True, exist_ok=True)
    out_figs = Path(args.out_figures); out_figs.mkdir(parents=True, exist_ok=True)

    # ----------------- LaTeX table -----------------
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-event LOEO results on PHEME (RoBERTa + CE). Reported are per-event Macro-F1 with test set size.}")
    lines.append(r"\label{tab:loeo_pheme_roberta_ce}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Event & \#Test & Macro-F1 $\uparrow$ \\")
    lines.append(r"\midrule")
    for _, rr in df.iterrows():
        e_tex = str(rr["event"]).replace("_", r"\_")
        lines.append(f"{e_tex} & {int(rr['n_test'])} & {float(rr['macro_f1']):.3f} \\\\")
    lines.append(r"\midrule")
    # summary for filtered set
    worst_event_tex = (str(worst_event).replace("_", r"\_")) if worst_event is not None else "N/A"
    lines.append(f"\\textbf{{Average (n\\_test $\geq$ {int(args.min_event_size)})}} & -- & \\textbf{{{avg_f1:.3f}}} \\\\")
    lines.append(f"\\textbf{{Worst (\\, {worst_event_tex} \\,)}} & -- & \\textbf{{{worst_f1:.3f}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)

    tex_path = out_tables / "loeo_pheme_roberta_ce.tex"
    tex_path.write_text(tex, encoding="utf-8")

    # ----------------- Figure: bar plot -----------------
    plt.figure(figsize=(10, 3.2))
    x = np.arange(len(df))
    plt.bar(x, df["macro_f1"].values)
    plt.xticks(x, df["event"].values, rotation=35, ha="right")
    plt.ylabel("Macro-F1")
    title = f"PHEME LOEO: RoBERTa + CE (per-event Macro-F1)"
    if args.min_event_size > 0:
        title += f" | summary uses n_test≥{args.min_event_size}"
    plt.title(title)
    plt.ylim(0, max(0.5, float(df["macro_f1"].max()) + 0.05))
    plt.tight_layout()

    fig_path = out_figs / "loeo_pheme_roberta_ce_macro_f1.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # ----------------- Excel output -----------------
    xlsx_path = out_tables / "loeo_pheme_roberta_ce.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="per_event")
        summary.to_excel(writer, index=False, sheet_name="summary")

    # optional CSVs (handy for quick diff / scripts)
    df.to_csv(out_tables / "loeo_pheme_roberta_ce.csv", index=False)
    summary.to_csv(out_tables / "loeo_pheme_roberta_ce_summary.csv", index=False)

    print("WROTE TABLE:", tex_path)
    print("WROTE FIG  :", fig_path)
    print("WROTE XLSX :", xlsx_path)
    print("AVG_F1 (filtered):", avg_f1)
    print("WORST_F1 (filtered):", worst_f1, "EVENT:", worst_event)


if __name__ == "__main__":
    main()
