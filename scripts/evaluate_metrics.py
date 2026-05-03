"""Evaluate model metrics for the concrete crack detection project.

This script computes dataset, classification, and CPU inference metrics when
the original dataset is available locally.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.inference import preprocess_image


def build_dataframe(dataset_dir: Path) -> pd.DataFrame:
    image_paths = sorted(dataset_dir.glob("*/*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG images found under {dataset_dir}")

    labels = [path.parent.name for path in image_paths]
    return pd.DataFrame({"JPG": [str(path) for path in image_paths], "CATEGORY": labels})


def label_to_int(label: str) -> int:
    mapping = {"Negative": 0, "Positive": 1}
    if label not in mapping:
        raise ValueError(f"Unsupported label: {label}")
    return mapping[label]


def predict_probabilities(model, image_paths: list[str]) -> list[float]:
    probabilities = []
    for image_path in image_paths:
        batch = preprocess_image(image_path)
        probabilities.append(float(model.predict(batch, verbose=0)[0][0]))
    return probabilities


def benchmark_model(model_path: Path, sample_image: Path, runs: int) -> dict:
    from tensorflow.keras.models import load_model

    process = psutil.Process(os.getpid())
    rss_before_mb = process.memory_info().rss / 1024 / 1024

    start = time.perf_counter()
    model = load_model(model_path)
    model_load_sec = time.perf_counter() - start

    rss_after_load_mb = process.memory_info().rss / 1024 / 1024
    batch = preprocess_image(str(sample_image))

    model.predict(batch, verbose=0)
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(batch, verbose=0)
        latencies.append(time.perf_counter() - start)

    rss_after_predict_mb = process.memory_info().rss / 1024 / 1024

    mean_latency_sec = statistics.mean(latencies)
    return {
        "model_load_sec": model_load_sec,
        "inference_latency_ms_mean": mean_latency_sec * 1000,
        "inference_latency_ms_median": statistics.median(latencies) * 1000,
        "throughput_images_per_sec": 1 / mean_latency_sec,
        "rss_before_load_mb": rss_before_mb,
        "rss_after_load_mb": rss_after_load_mb,
        "rss_after_predict_mb": rss_after_predict_mb,
        "model": model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate crack detection metrics")
    parser.add_argument("--dataset", type=Path, default=Path("../input/surface-crack-detection"))
    parser.add_argument("--model", type=Path, default=Path("Model_Last_Prediction.h5"))
    parser.add_argument("--output", type=Path, default=Path("docs/metrics_latest.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    sample_image = Path("/tmp/cnn_metrics_sample.jpg")
    cv2.imwrite(str(sample_image), np.full((200, 200, 3), 128, dtype=np.uint8))
    benchmark = benchmark_model(args.model, sample_image, args.runs)
    model = benchmark.pop("model")

    result = {
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "gpu_count": 0,
        },
        "deployment": benchmark,
    }

    if args.dataset.exists():
        data = build_dataframe(args.dataset)
        train_df, test_df = train_test_split(data, train_size=0.9, shuffle=True, random_state=42)
        train_generator_count = int(len(train_df) * 0.9)
        validation_generator_count = len(train_df) - train_generator_count

        y_true = [label_to_int(label) for label in test_df["CATEGORY"]]
        y_score = predict_probabilities(model, test_df["JPG"].tolist())
        y_pred = [1 if score >= args.threshold else 0 for score in y_score]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        result["dataset"] = {
            "total_images": len(data),
            "class_distribution": data["CATEGORY"].value_counts().to_dict(),
            "train_images": train_generator_count,
            "validation_images": validation_generator_count,
            "test_images": len(test_df),
        }
        result["performance"] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_score),
            "log_loss": log_loss(y_true, y_score),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_negative": int(tn),
            "error_rate": 1 - accuracy_score(y_true, y_pred),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else None,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) else None,
            "detection_rate": recall_score(y_true, y_pred, zero_division=0),
        }
    else:
        result["dataset_error"] = f"Dataset path not found: {args.dataset}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
