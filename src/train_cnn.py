"""Reference CNN training script based on the 2024 course project.

This is a cleaned-up, standalone implementation of the model architecture
from the original final scripts. It was NOT used to generate the reported
historical accuracy tables in docs/RESULTS.md.

Usage:
  python src/train_cnn.py --data-dir data --epochs 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["DC", "Marvel"]


def load_dataset(path: Path, shuffle: bool):
    if not path.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {path}")
    return keras.utils.image_dataset_from_directory(
        path,
        class_names=CLASS_NAMES,
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )


def build_model() -> keras.Model:
    """Portable model structurally matching the final 2024 CNN scripts.

    3 convolution/max-pooling pairs, a 512-neuron dense layer and sigmoid output.
    No dropout or augmentation were used in the supplied final CNN scripts.
    """
    return keras.Sequential(
        [
            keras.Input(shape=(*IMAGE_SIZE, 3)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    tf.keras.utils.set_random_seed(SEED)
    # Only the training dataset is shuffled. Evaluation datasets have stable order.
    train = load_dataset(args.data_dir / "train", shuffle=True)
    valid = load_dataset(args.data_dir / "valid", shuffle=False)
    test = load_dataset(args.data_dir / "test", shuffle=False)

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=valid, epochs=args.epochs)
    loss, accuracy = model.evaluate(test, verbose=0, return_dict=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "cnn_reference.keras")
    summary = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "epochs": args.epochs,
        "classes": CLASS_NAMES,
        "note": "New reference run; not a reproduction of reported 2024 results.",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
