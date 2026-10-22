"""Ablation study for the custom CNN: add one change per run, each on top of the previous one.

    0 baseline      Flatten -> Dense(512) head, 1 conv per block, 40 epochs, LR-on-plateau, 160px
    1 +gap          GlobalAveragePooling2D head instead of Flatten -> Dense(512)
    2 +longer       80 epochs with a cosine LR schedule (and more patience for early stopping)
    3 +2conv        two 3x3 convs per block
    4 +224px        224x224 input

Decide whether to keep a change by its validation numbers. The test numbers are reported for every
step, but choosing steps by test accuracy would turn the test set into a second validation set.
The McNemar column compares each step with the one before it on the same test images.

Finished steps (with eval/metrics.json) are skipped, so the script can be rerun after a Colab
disconnect and picks up where it stopped.

Usage
    python -m catdog.ablation --data data --out runs/ablation --report reports/ablation.md
    python -m catdog.ablation --steps 0 1          # only the first two runs
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import keras
import pandas as pd

from catdog.metrics import mcnemar

BASE = ["--model", "cnn", "--epochs", "40", "--patience", "5", "--image-size", "160",
        "--head", "flatten", "--convs-per-block", "1", "--schedule", "plateau"]

# Each step overrides some flags of the step before it.
STEPS = [
    ("baseline", "original CNN", {}),
    ("gap", "+ GlobalAveragePooling2D head", {"--head": "gap"}),
    ("longer", "+ 80 epochs, cosine LR", {"--epochs": "80", "--schedule": "cosine", "--patience": "15"}),
    ("2conv", "+ 2 convs per block", {"--convs-per-block": "2"}),
    ("224px", "+ 224x224 input", {"--image-size": "224"}),
]


def step_args():
    """Cumulative flags for every step."""
    flags = dict(zip(BASE[::2], BASE[1::2]))
    out = []
    for name, label, change in STEPS:
        flags = {**flags, **change}
        out.append((name, label, [x for kv in flags.items() for x in kv]))
    return out


def run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True)


def summarize(run_dir):
    m = json.loads((run_dir / "eval" / "metrics.json").read_text())
    h = pd.read_csv(run_dir / "history.csv")
    best = h.loc[h["val_loss"].idxmin()]
    return {
        "params": keras.models.load_model(run_dir / "model.keras").count_params(),
        "epochs run": len(h),
        "best epoch": int(best["epoch"]) + 1,
        "train acc": best["accuracy"],
        "val acc": best["val_accuracy"],
        "val loss": best["val_loss"],
        "test acc": m["at_0.5"]["accuracy"],
        "test 95% CI": "{:.3f}-{:.3f}".format(*m["ci95"]["accuracy"]),
        "dog recall": m["at_0.5"]["recall_dog"],
        "ECE": m["at_0.5"]["ece"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="runs/ablation")
    parser.add_argument("--report", default="reports/ablation.md")
    parser.add_argument("--steps", type=int, nargs="+", help="step numbers to run (default: all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, help="cap every step's epochs (for a quick smoke test)")
    args = parser.parse_args()

    steps = step_args()
    todo = args.steps if args.steps is not None else range(len(steps))
    out = Path(args.out)
    for i in todo:
        name, label, flags = steps[i]
        run_dir = out / f"{i}_{name}"
        if args.max_epochs:
            e = flags.index("--epochs") + 1
            flags[e] = str(min(int(flags[e]), args.max_epochs))
        if (run_dir / "eval" / "metrics.json").exists():
            print(f"[{i}] {label}: already done, skipping")
            continue
        print(f"[{i}] {label}: python -m catdog.train {' '.join(flags)}")
        run("catdog.train", *flags, "--data", args.data, "--out", str(run_dir),
            "--batch-size", str(args.batch_size), "--seed", str(args.seed))
        run("catdog.evaluate", "--run", str(run_dir), "--data", args.data, "--n-boot", str(args.n_boot))

    # Report every finished step, in order, with a paired test against the previous finished step.
    rows, prev = [], None
    for i, (name, label, _) in enumerate(steps):
        run_dir = out / f"{i}_{name}"
        if not (run_dir / "eval" / "metrics.json").exists():
            continue
        row = {"step": i, "change": label, **summarize(run_dir)}
        preds = pd.read_csv(run_dir / "eval" / "predictions_test.csv").sort_values("path")
        if prev is not None:
            assert (prev["path"].values == preds["path"].values).all(), "runs used different test sets"
            t = mcnemar(preds["label"].values, (prev["p_dog"].values >= 0.5).astype(int),
                        (preds["p_dog"].values >= 0.5).astype(int))
            row["vs previous (fixed / broken, p)"] = (
                f"{t['only_b_correct']} / {t['only_a_correct']}, p = {t['p_value']:.2g}")
        rows.append(row)
        prev = preds

    table = pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")
    report = ["## Custom CNN ablation", "",
              "Each row adds one change to the row above. Keep or drop a change based on val acc / val loss; "
              "test numbers are for reporting only. 'fixed / broken' counts test images the step got right "
              "that the previous step got wrong, and the reverse (exact McNemar test).", "", table, ""]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
