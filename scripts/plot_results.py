import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_metrics(run_dir):
    with open(run_dir / "summary.json", "r") as f:
        return json.load(f)


def plot_bar(metrics, out_dir):
    names = ["accuracy", "macro_f1", "ece"]
    values = [metrics[n] for n in names]

    plt.figure()
    plt.bar(names, values)
    plt.title("Performance Summary")
    plt.ylabel("Score")
    plt.savefig(out_dir / "bar_performance.png")
    plt.close()


def plot_reliability(run_dir, out_dir):
    # Load predictions
    preds_file = run_dir / "predictions_test.jsonl"

    probs = []
    labels = []

    with open(preds_file, "r") as f:
        for line in f:
            row = json.loads(line)
            probs.append(row["probs"])
            labels.append(row["label"])

    probs = np.array(probs)
    labels = np.array(labels)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    bins = np.linspace(0, 1, 11)
    bin_acc = []
    bin_conf = []

    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            acc = (predictions[mask] == labels[mask]).mean()
            conf = confidences[mask].mean()
            bin_acc.append(acc)
            bin_conf.append(conf)
        else:
            bin_acc.append(0)
            bin_conf.append(0)

    plt.figure()
    plt.plot(bin_conf, bin_acc, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.savefig(out_dir / "reliability.png")
    plt.close()


def main():
    run_name = "pheme_roberta_ce_indomain_seed42"  # CHANGE THIS

    run_dir = ROOT / "outputs" / "runs" / run_name
    out_dir = ROOT / "outputs" / "figures" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(run_dir)

    plot_bar(metrics, out_dir)
    plot_reliability(run_dir, out_dir)

    print("FIGURES GENERATED:", out_dir)


if __name__ == "__main__":
    main()