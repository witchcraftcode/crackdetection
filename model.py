"""Prediction helper used by the FastAPI app."""

import os
from functools import lru_cache

from src.inference import load_keras_model, predict_with_model


DEFAULT_MODEL_PATH = "Model_Last_Prediction.h5"


@lru_cache(maxsize=1)
def get_model(model_path: str = DEFAULT_MODEL_PATH):
    return load_keras_model(model_path)


def predict(img_path: str, threshold: float = 0.5):
    """Predict crack presence for a given image path."""
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    label, score = predict_with_model(get_model(model_path), img_path, threshold=threshold)
    return {"label": label, "score": score}
