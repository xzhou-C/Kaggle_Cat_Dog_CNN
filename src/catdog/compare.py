"""Compare evaluated runs on the same test set and write a markdown results table.

Includes an exact McNemar test for each pair of models, which asks whether the difference in
errors is bigger than chance on these specific test images (a paired test, unlike comparing two
accuracies with separate CIs).

Usage
    python -m catdog.compare runs/cnn runs/effnet --out reports/results.md
"""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from catdog.metrics import mcnemar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="reports/results.md")
    args = parser.parse_args()

    rows, preds = [], {}
    for run in map(Path, args.runs):
        m = json.loads((run / "eval" / "metrics.json").read_text())
        a, ci = m["at_0.5"], m["ci95"]
        rows.append({
            "model": m["model"],
            "test accuracy (95% CI)": f"{a['accuracy']:.3f} ({ci['accuracy'][0]:.3f}-{ci['accuracy'][1]:.3f})",
            "F1": f"{a['f1']:.3f}",
            "ROC AUC": f"{a['roc_auc']:.4f}",
            "cat recall": f"{a['recall_cat']:.3f}",
            "dog recall": f"{a['recall_dog']:.3f}",
            "log loss": f"{a['log_loss']:.3f}",
            "ECE": f"{a['ece']:.3f}",
        })
        preds[m["model"]] = pd.read_csv(run / "eval" / "predictions_test.csv").sort_values("path")

    lines = ["## Test-set results", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    for a, b in itertools.combinations(preds, 2):
        pa, pb = preds[a], preds[b]
        assert (pa["path"].values == pb["path"].values).all(), "runs were evaluated on different test sets"
        y = pa["label"].values
        t = mcnemar(y, (pa["p_dog"].values >= 0.5).astype(int), (pb["p_dog"].values >= 0.5).astype(int))
        lines += [f"**McNemar {a} vs {b}:** {t['only_a_correct']} images only {a} got right, "
                  f"{t['only_b_correct']} only {b} got right, p = {t['p_value']:.2g}.", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
