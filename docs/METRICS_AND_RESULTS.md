# Deep Learning-Based Crack Detection System Metrics and Results

This document is the tracked metrics record for the repository. Values are based on the saved notebook `risk-flag.ipynb`, the saved local model artifact `Model_Last_Prediction.h5`, and local CPU benchmarking performed from the project `.venv`.

Last updated: 2026-05-03.

## Dataset Metrics

| Metric | Result |
|---|---:|
| Total number of images | 40,000 |
| Train / validation / test split ratio | 81% / 9% / 10% |
| Train images | 32,400 |
| Validation images | 3,600 |
| Test images | 4,000 |
| Class distribution | 2 classes, `Negative` and `Positive`; notebook dataset appears balanced |
| Number of classes | 2 |
| Image resolution | 200 x 200 x 3 |
| Dataset size after augmentation | Online augmentation during training; no fixed expanded dataset saved |
| Augmentation techniques count | 7 image-changing techniques plus rescaling |
| Augmentation techniques | rotation, shear, zoom, width shift, height shift, brightness, vertical flip, rescale |

## Model Architecture Metrics

| Metric | Result |
|---|---:|
| Number of layers | 16 including input, convolution, batch normalization, pooling, dropout, flatten, and dense layers |
| Total number of parameters | 38,139,331 |
| Trainable parameters | 38,139,265 |
| Non-trainable parameters | 64 |
| Model file size | 437 MB |
| In-memory parameter size | 145.49 MB |
| Kernel sizes | 3 x 3 convolution kernels |
| Activation functions used | ReLU, sigmoid |
| Pooling type | MaxPooling2D |
| Input shape | 200 x 200 x 3 |

## Training Metrics

| Metric | Result |
|---|---:|
| Number of epochs configured | 50 |
| Number of epochs completed | 19 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer used | Adam |
| Loss function | Binary cross-entropy |
| Training time | 7,776 seconds / 129.6 minutes / 2.16 hours |
| Hardware used | Not recorded during original notebook training |

## Performance Metrics

| Metric | Result |
|---|---:|
| Test accuracy | 89.88% |
| Log loss / test loss | 0.2376 |
| Precision | Requires original dataset files; compute with `scripts/evaluate_metrics.py` |
| Recall | Requires original dataset files; compute with `scripts/evaluate_metrics.py` |
| F1-score | Requires original dataset files; compute with `scripts/evaluate_metrics.py` |
| ROC-AUC | Requires original dataset files; compute with `scripts/evaluate_metrics.py` |
| Confusion matrix values | Requires original dataset files; compute with `scripts/evaluate_metrics.py` |

## Generalization and Regularization Metrics

| Metric | Result |
|---|---:|
| Final training accuracy | 99.07% |
| Final validation accuracy | 97.42% |
| Final training loss | 0.0331 |
| Final validation loss | 0.0705 |
| Final generalization gap | 1.65 percentage points |
| Best validation accuracy | 99.00% at epoch 8 |
| Dropout rate | 0.3, 0.3, 0.3, 0.5 |
| Early stopping epoch | 19 |
| Overfitting reduction methods | Data augmentation, dropout, batch normalization, early stopping |

## Inference and Deployment Metrics

Measured locally on macOS arm64 with Python 3.10, TensorFlow 2.21.0, CPU only, 8 CPU cores, and no detected GPU.

| Metric | Result |
|---|---:|
| Mean inference latency per image | 32.49 ms |
| Median inference latency per image | 32.49 ms |
| Throughput | 30.78 images per second |
| Model loading time | 0.14 seconds |
| Memory before model load | 501 MB RSS |
| Memory after model load | 949 MB RSS |
| Memory after prediction | 968 MB RSS |
| CPU inference time | 32.49 ms per image |
| GPU inference time | No GPU detected; not applicable in this environment |

The raw machine-readable run is stored in `docs/metrics_latest.json`.

## Impact Metrics

| Metric | Result |
|---|---:|
| Error rate | 10.12% |
| Detection rate | Requires confusion matrix/recall computation from original dataset files |
| False positive rate | Requires original dataset files; compute with `scripts/evaluate_metrics.py` |
| False negative rate | Requires original dataset files; compute with `scripts/evaluate_metrics.py` |
| Improvement over baseline | Baseline model was not recorded in the notebook |
| Reduction in manual inspection time | Not measured; requires a timed manual-inspection study |

## Reproducible Measurement Command

When the original dataset is available at `../input/surface-crack-detection`, run:

```bash
source .venv/bin/activate
python scripts/evaluate_metrics.py \
  --dataset ../input/surface-crack-detection \
  --model Model_Last_Prediction.h5 \
  --output docs/metrics_latest.json
```

The script computes precision, recall, F1-score, ROC-AUC, confusion matrix values, false positive rate, false negative rate, detection rate, CPU latency, throughput, model loading time, and memory usage.
