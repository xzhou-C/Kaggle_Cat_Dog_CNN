"""Clean the raw Microsoft/Kaggle PetImages folder and write a stratified train/val/test split.

Steps
1. Skip files that are not decodable images (Thumbs.db, truncated JPEGs, 0-byte files).
2. Skip images flagged during manual review (corrupted or mislabeled, see MANUAL_EXCLUDE).
3. Drop exact duplicates (same pixel content), so the same photo cannot land in both train and test.
4. Re-encode everything as RGB JPEG (fixes grayscale, CMYK, RGBA and PNG-in-.jpg files).
5. Split each class 80/10/10 with a fixed seed and write a manifest.csv describing every kept file.

Usage
    python -m catdog.prepare_data --src /path/to/PetImages --out data
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from catdog import CLASS_NAMES

# Image ids found during manual review of the raw dataset (corrupted or not a cat/dog).
MANUAL_EXCLUDE = {
    "Dog": {5604, 6413, 8736, 8898, 9188, 9517, 10161, 10190, 10237, 10401, 10797, 11186, 11702},
    "Cat": {666, 2939, 3216, 3672, 4688, 4833, 5418, 6215, 7377, 8456, 8470, 10029, 11565, 12272},
}


def load_clean(path):
    """Return an RGB PIL image, or None if the file is not a usable image."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:  # verify() leaves the image unusable, so reopen to decode
            img.load()
            return img.convert("RGB")  # convert() returns a new image, safe to use after close
    except Exception:
        return None


def collect(src, log=print):
    rows = []
    seen = {}
    for class_name in CLASS_NAMES:
        files = sorted((Path(src) / class_name).iterdir())
        n_bad = n_excluded = n_dup = 0
        for f in files:
            if f.stem.isdigit() and int(f.stem) in MANUAL_EXCLUDE[class_name]:
                n_excluded += 1
                continue
            img = load_clean(f)
            if img is None:
                n_bad += 1
                continue
            digest = hashlib.md5(img.tobytes()).hexdigest()
            if digest in seen:
                n_dup += 1
                continue
            seen[digest] = f
            rows.append({"source": str(f), "label": class_name, "width": img.width,
                         "height": img.height, "md5": digest})
        log(f"{class_name}: {len(files)} files, {n_bad} unreadable, {n_excluded} manually excluded, "
            f"{n_dup} duplicates, {sum(r['label'] == class_name for r in rows)} kept")
    return pd.DataFrame(rows)


def assign_splits(df, train=0.8, val=0.1, seed=41):
    """Stratified train/val/test split: every split keeps the same Cat/Dog ratio."""
    train_idx, rest_idx = train_test_split(
        df.index, train_size=train, stratify=df["label"], random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val / (1 - train), stratify=df.loc[rest_idx, "label"], random_state=seed)
    df = df.copy()
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def write_split(df, out):
    out = Path(out)
    paths = []
    for row in df.itertuples():
        dest = out / row.split / row.label / f"{row.label.lower()}_{Path(row.source).stem}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        load_clean(row.source).save(dest, "JPEG", quality=95)
        paths.append(str(dest.relative_to(out)))
    df = df.assign(path=paths)
    df.drop(columns=["source"]).to_csv(out / "manifest.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="PetImages folder containing Cat/ and Dog/")
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    df = assign_splits(collect(args.src), seed=args.seed)
    write_split(df, args.out)
    print(df.groupby(["split", "label"]).size().unstack())


if __name__ == "__main__":
    main()
