"""Train a model and save the best checkpoint (lowest val loss) plus its training history.

Usage
    python -m catdog.train --model cnn    --data data --out runs/cnn    --epochs 40
    python -m catdog.train --model effnet --data data --out runs/effnet --epochs 5 --finetune-epochs 10
"""
import argparse
import json
from pathlib import Path

import keras

from catdog.data import load_all
from catdog.models import MODELS, unfreeze_top


def callbacks(out, patience, plateau=True):
    cbs = [
        keras.callbacks.ModelCheckpoint(out / "model.keras", monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.CSVLogger(out / "history.csv", append=True),
    ]
    if plateau:  # a cosine schedule sets the LR itself, so it can't be combined with LR-on-plateau
        cbs.insert(2, keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7))
    return cbs


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--schedule", choices=["plateau", "cosine"], default="plateau",
                        help="halve the LR when val loss stalls, or cosine-decay it to 0 over --epochs")
    parser.add_argument("--head", choices=["flatten", "gap"], default="flatten", help="cnn only")
    parser.add_argument("--convs-per-block", type=int, default=1, help="cnn only")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="effnet only")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="effnet only")
    parser.add_argument("--finetune-layers", type=int, default=40, help="effnet only")
    parser.add_argument("--weights", default="imagenet", help="effnet only; 'none' for random init")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.csv").unlink(missing_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    size = (args.image_size, args.image_size)
    train, val, _, _ = load_all(args.data, size, args.batch_size, args.seed)

    if args.model == "effnet":
        weights = None if args.weights.lower() == "none" else args.weights
        model = MODELS["effnet"](size, weights=weights)
    else:
        model = MODELS["cnn"](size, head=args.head, convs_per_block=args.convs_per_block)

    # Stage 1: train the whole CNN, or only the new head on top of the frozen backbone.
    cosine = args.schedule == "cosine"
    lr = keras.optimizers.schedules.CosineDecay(args.lr, args.epochs * int(train.cardinality())) if cosine else args.lr
    compile_model(model, lr)
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks(out, args.patience, plateau=not cosine))

    # Stage 2 (transfer learning only): fine-tune the top of the backbone with a small learning rate.
    if args.model == "effnet" and args.finetune_epochs > 0:
        model.set_weights(keras.models.load_model(out / "model.keras").get_weights())
        stage1_best = model.evaluate(val, verbose=0)[0]
        unfreeze_top(model, args.finetune_layers)
        compile_model(model, args.finetune_lr)
        cbs = callbacks(out, args.patience)
        cbs[0].best = stage1_best  # only overwrite the checkpoint if fine-tuning actually helps
        model.fit(train, validation_data=val, epochs=args.finetune_epochs, callbacks=cbs)

    print(f"Best model saved to {out / 'model.keras'}")


if __name__ == "__main__":
    main()
