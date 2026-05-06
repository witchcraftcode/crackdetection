"""CLI inference for binary crack detection model."""

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LABELS = {0: "Negative", 1: "Positive"}


def validate_model_artifact(model_path: str) -> None:
    """Fail early with a useful message when the model artifact is missing."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Place Model_Last_Prediction.h5 in the project root or pull Git LFS files with `git lfs pull`."
        )

    with path.open("rb") as file:
        header = file.read(128)

    if header.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"{model_path} is a Git LFS pointer, not the actual model file. "
            "Install Git LFS and run `git lfs pull` to download the trained model."
        )


def preprocess_image(image_path: str, target_size=(200, 200)) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def load_keras_model(model_path: str) -> Any:
    """Load a Keras model lazily so imports stay lightweight and testable."""
    validate_model_artifact(model_path)
    from tensorflow.keras.models import load_model

    return load_model(model_path)


def predict_with_model(model: Any, image_path: str, threshold: float = 0.5):
    image_batch = preprocess_image(image_path)

    probability = float(model.predict(image_batch, verbose=0)[0][0])
    klass = 1 if probability >= threshold else 0
    return LABELS[klass], probability


def predict_single(model_path: str, image_path: str, threshold: float = 0.5):
    model = load_keras_model(model_path)
    return predict_with_model(model, image_path, threshold=threshold)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict crack presence from a single concrete image")
    parser.add_argument("--model", required=True, help="Path to a Keras .h5/.keras model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for Positive class")
    return parser.parse_args()


def main():
    args = parse_args()
    label, probability = predict_single(args.model, args.image, threshold=args.threshold)
    print(f"label={label} probability={probability:.4f}")


if __name__ == "__main__":
    main()
