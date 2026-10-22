"""Evaluate a trained run on the held-out test set.

Writes to <run>/eval/:
    predictions_val.csv, predictions_test.csv   one row per image: path, label, p_dog
    metrics.json        point metrics at 0.5 and at the val-tuned threshold, with bootstrap 95% CIs
    slices.csv          test accuracy by image resolution, aspect ratio and brightness
    confusion.png, roc.png, calibration.png, errors_gradcam.png

Usage
    python -m catdog.evaluate --run runs/effnet --data data
"""
import argparse
import json
import os
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve

from catdog import CLASS_NAMES
from catdog.data import load_split
from catdog.gradcam import gradcam, overlay
from catdog.metrics import best_threshold, bootstrap_ci, point_metrics


def predict(model, data_dir, split, image_size, batch_size):
    ds, paths = load_split(data_dir, split, image_size, batch_size)
    y = np.concatenate([yb.numpy().ravel() for _, yb in ds]).astype(int)
    p = model.predict(ds, verbose=0).ravel()
    rel = [os.path.relpath(f, data_dir) for f in paths]  # same form as manifest.csv: test/Cat/cat_1.jpg
    return pd.DataFrame({"path": rel, "label": y, "p_dog": p})


def add_image_features(df, data_dir):
    """Join original resolution from the manifest and compute mean brightness per image."""
    manifest = Path(data_dir) / "manifest.csv"
    if manifest.exists():
        df = df.merge(pd.read_csv(manifest)[["path", "width", "height"]], on="path", how="left")
    else:
        sizes = [Image.open(Path(data_dir) / f).size for f in df["path"]]
        df["width"], df["height"] = zip(*sizes)
    df["brightness"] = [np.asarray(Image.open(Path(data_dir) / f).convert("L")).mean() for f in df["path"]]
    df["min_side"] = df[["width", "height"]].min(axis=1)
    df["aspect"] = df["width"] / df["height"]
    return df


def slice_table(df, threshold):
    df = df.assign(correct=((df["p_dog"] >= threshold).astype(int) == df["label"]))
    slices = {
        "min_side_px": pd.cut(df["min_side"], [0, 150, 250, 350, 10_000],
                              labels=["<150", "150-250", "250-350", ">350"]),
        "aspect": pd.cut(df["aspect"], [0, 0.8, 1.25, 100], labels=["portrait", "square-ish", "landscape"]),
        "brightness": pd.qcut(df["brightness"], 4, labels=["darkest 25%", "q2", "q3", "brightest 25%"],
                              duplicates="drop"),
        "class": df["label"].map(dict(enumerate(CLASS_NAMES))),
    }
    rows = []
    for name, groups in slices.items():
        for group, sub in df.groupby(groups, observed=True):
            n, acc = len(sub), sub["correct"].mean()
            se = np.sqrt(acc * (1 - acc) / n) if n else np.nan
            rows.append({"slice": name, "group": group, "n": n, "accuracy": acc,
                         "ci95_low": max(0, acc - 1.96 * se), "ci95_high": min(1, acc + 1.96 * se)})
    return pd.DataFrame(rows)


def plot_confusion(y, pred, path):
    cm = skm.confusion_matrix(y, pred, labels=[0, 1])
    skm.ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues", colorbar=False)
    plt.title("Test confusion matrix")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc(y, p, path):
    skm.RocCurveDisplay.from_predictions(y, p, name="P(Dog)")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.title("Test ROC curve")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(y, p, path, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfectly calibrated")
    plt.plot(mean_pred, frac_pos, "o-", label="model")
    plt.xlabel("Predicted P(Dog)")
    plt.ylabel("Observed fraction of dogs")
    plt.title("Reliability diagram (test)")
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_errors(model, df, data_dir, image_size, threshold, path, n=12):
    """The model's most confident mistakes, with Grad-CAM overlays."""
    wrong = df[(df["p_dog"] >= threshold).astype(int) != df["label"]].copy()
    wrong["confidence"] = np.maximum(wrong["p_dog"], 1 - wrong["p_dog"])
    wrong = wrong.nlargest(n, "confidence")
    if wrong.empty:
        return
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    plt.figure(figsize=(cols * 3, rows * 3.3))
    for i, row in enumerate(wrong.itertuples()):
        img = np.asarray(Image.open(Path(data_dir) / row.path).convert("RGB").resize(image_size), dtype=np.float32)
        heat, p = gradcam(model, img)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(overlay(img, heat))
        plt.title(f"true {CLASS_NAMES[row.label]}, P(Dog)={p:.2f}", fontsize=9)
        plt.axis("off")
    plt.suptitle("Most confident test errors (Grad-CAM)")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    run = Path(args.run)
    out = run / "eval"
    out.mkdir(exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    size = (config["image_size"],) * 2
    model = keras.models.load_model(run / "model.keras")

    val = predict(model, args.data, "val", size, args.batch_size)
    test = predict(model, args.data, "test", size, args.batch_size)
    val.to_csv(out / "predictions_val.csv", index=False)

    threshold = best_threshold(val["label"].values, val["p_dog"].values)
    y, p = test["label"].values, test["p_dog"].values
    acc = lambda t: (lambda yy, pp: skm.accuracy_score(yy, (pp >= t).astype(int)))
    results = {
        "model": config["model"],
        "n_test": int(len(test)),
        "tuned_threshold_from_val": threshold,
        "at_0.5": point_metrics(y, p, 0.5),
        "at_tuned": point_metrics(y, p, threshold),
        "ci95": {
            "accuracy": bootstrap_ci(y, p, acc(0.5), args.n_boot),
            "f1": bootstrap_ci(y, p, lambda yy, pp: skm.f1_score(yy, (pp >= 0.5).astype(int)), args.n_boot),
            "roc_auc": bootstrap_ci(y, p, skm.roc_auc_score, args.n_boot),
        },
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    test = add_image_features(test, args.data)
    test.to_csv(out / "predictions_test.csv", index=False)
    slice_table(test, 0.5).to_csv(out / "slices.csv", index=False)

    plot_confusion(y, (p >= 0.5).astype(int), out / "confusion.png")
    plot_roc(y, p, out / "roc.png")
    plot_calibration(y, p, out / "calibration.png")
    plot_errors(model, test, args.data, size, 0.5, out / "errors_gradcam.png")

    m = results["at_0.5"]
    lo, hi = results["ci95"]["accuracy"]
    print(f"{config['model']}: test accuracy {m['accuracy']:.4f} (95% CI {lo:.4f}-{hi:.4f}), "
          f"AUC {m['roc_auc']:.4f}, ECE {m['ece']:.4f}. Outputs in {out}")


if __name__ == "__main__":
    main()
