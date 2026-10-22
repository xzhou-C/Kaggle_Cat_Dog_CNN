"""tf.data pipelines.

Images are loaded once at the target size and cached as uint8. Augmentation and pixel
scaling live inside the model (see models.py), so:
- augmentation is re-sampled every epoch (caching can't freeze it), and
- train, val, test and the demo app all go through exactly the same resize path.
"""
from pathlib import Path

import tensorflow as tf

from catdog import CLASS_NAMES


def load_split(data_dir, split, image_size=(160, 160), batch_size=32, shuffle=False, seed=1):
    ds = tf.keras.utils.image_dataset_from_directory(
        Path(data_dir) / split,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=image_size,
        batch_size=None,
        shuffle=False,  # keep file order so predictions line up with ds.file_paths
    )
    file_paths = ds.file_paths
    ds = ds.map(lambda x, y: (tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8), y),
                num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle:
        # Files arrive sorted by class (all cats, then all dogs), so the buffer must hold the whole
        # split. A smaller buffer gives single-class batches and a CNN that never converges.
        ds = ds.shuffle(len(file_paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, file_paths


def load_all(data_dir, image_size=(160, 160), batch_size=32, seed=1):
    train, _ = load_split(data_dir, "train", image_size, batch_size, shuffle=True, seed=seed)
    val, val_paths = load_split(data_dir, "val", image_size, batch_size)
    test, test_paths = load_split(data_dir, "test", image_size, batch_size)
    return train, val, test, {"val": val_paths, "test": test_paths}
