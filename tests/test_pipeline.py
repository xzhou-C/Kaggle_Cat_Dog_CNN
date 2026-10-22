"""End-to-end smoke test on a tiny synthetic dataset (runs on CPU in about a minute)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from catdog.metrics import best_threshold, expected_calibration_error, mcnemar, point_metrics

ROOT = Path(__file__).resolve().parents[1]


def make_raw(root, n=40, seed=0):
    """Fake PetImages: cats are reddish, dogs bluish, plus the junk the real dataset contains."""
    rng = np.random.default_rng(seed)
    for cls, channel in [("Cat", 0), ("Dog", 2)]:
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            arr = rng.integers(0, 100, (rng.integers(60, 120), rng.integers(60, 120), 3), dtype=np.uint8)
            arr[..., channel] += 120
            Image.fromarray(arr).save(d / f"{i}.jpg")
        (d / "Thumbs.db").write_bytes(b"not an image")
        (d / f"{n}.jpg").write_bytes(b"\xff\xd8 truncated")
        shutil.copy(d / "0.jpg", d / f"{n + 1}.jpg")  # exact duplicate of 0.jpg
        Image.open(d / "1.jpg").convert("L").save(d / f"{n + 2}.jpg")  # grayscale
    return root


def run(*args, cwd):
    subprocess.run([sys.executable, "-m", *args], cwd=cwd, check=True,
                   env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "TF_CPP_MIN_LOG_LEVEL": "2"})


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    wd = tmp_path_factory.mktemp("work")
    make_raw(wd / "PetImages")
    run("catdog.prepare_data", "--src", "PetImages", "--out", "data", cwd=wd)
    return wd


def test_prepare_data_cleans_and_splits(workdir):
    m = pd.read_csv(workdir / "data" / "manifest.csv")
    assert len(m) == 2 * 41  # 40 originals + 1 grayscale per class; junk and duplicates dropped
    assert m["md5"].is_unique
    assert set(m["split"]) == {"train", "val", "test"}
    for f in m["path"]:
        assert Image.open(workdir / "data" / f).mode == "RGB"


@pytest.mark.parametrize("model,extra", [
    ("cnn", []),
    ("effnet", ["--weights", "none", "--finetune-epochs", "1"]),
])
def test_train_and_evaluate(workdir, model, extra):
    out = f"runs/{model}"
    run("catdog.train", "--model", model, "--out", out, "--epochs", "2", "--image-size", "64",
        "--batch-size", "8", *extra, cwd=workdir)
    run("catdog.evaluate", "--run", out, "--n-boot", "50", cwd=workdir)
    ev = workdir / out / "eval"
    metrics = json.loads((ev / "metrics.json").read_text())
    assert 0 <= metrics["at_0.5"]["accuracy"] <= 1
    for f in ["confusion.png", "roc.png", "calibration.png", "slices.csv", "predictions_test.csv"]:
        assert (ev / f).exists(), f


def test_compare(workdir):
    if not all((workdir / "runs" / m / "eval" / "metrics.json").exists() for m in ["cnn", "effnet"]):
        pytest.skip("training tests did not run")
    run("catdog.compare", "runs/cnn", "runs/effnet", "--out", "results.md", cwd=workdir)
    assert "McNemar" in (workdir / "results.md").read_text()


def test_metrics_units():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = point_metrics(y, p)
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0
    assert expected_calibration_error(np.array([1, 1]), np.array([1.0, 1.0])) == 0
    assert 0.4 <= best_threshold(y, p) <= 0.6
    t = mcnemar(y, np.array([0, 0, 1, 1]), np.array([1, 1, 1, 1]))
    assert t["only_a_correct"] == 2 and t["only_b_correct"] == 0



def test_train_split_is_fully_shuffled(tmp_path):
    """Files load sorted by class (all cats, then all dogs). With 5k images per class, a shuffle
    buffer smaller than the split leaves the start of each epoch almost all cats."""
    from catdog.data import load_split
    tiny = np.zeros((8, 8, 3), np.uint8)
    for cls in ["Cat", "Dog"]:
        d = tmp_path / "train" / cls
        d.mkdir(parents=True)
        for i in range(5000):
            Image.fromarray(tiny).save(d / f"{i}.png")
    ds, _ = load_split(tmp_path, "train", (8, 8), batch_size=2500, shuffle=True)
    first_quarter = next(iter(ds))[1].numpy()
    assert 0.4 < first_quarter.mean() < 0.6


def test_ablation(workdir):
    run("catdog.ablation", "--max-epochs", "1", "--n-boot", "20", "--batch-size", "8",
        "--out", "runs/ablation", "--report", "ablation.md", cwd=workdir)
    report = (workdir / "ablation.md").read_text()
    for label in ["original CNN", "GlobalAveragePooling2D", "cosine", "2 convs", "224x224"]:
        assert label in report, label
