# Concrete Surface Crack Classification (CNN)

CNN-based binary classifier for concrete surface crack detection.

## Overview
This project predicts whether a concrete surface image contains a crack:
- `Positive`: crack present
- `Negative`: no visible crack

The current training workflow is in the notebook `risk-flag.ipynb`, and reusable code is available in `src/`.

## Project Structure
```text
.
├── app.py                         # FastAPI service for image upload prediction
├── model.py                       # Cached model-loading helper used by the API
├── Model_Last_Prediction.h5       # Trained Keras model stored with Git LFS
├── src/
│   ├── model.py                   # Reusable CNN model builder
│   └── inference.py               # Preprocessing and single-image prediction logic
├── scripts/
│   └── evaluate_metrics.py        # Dataset, classification, and inference metrics script
├── tests/
│   └── test_imports_and_preprocess.py
├── docs/
│   ├── METRICS_AND_RESULTS.md     # Human-readable metrics report
│   └── metrics_latest.json        # Latest machine-readable metrics output
├── risk-flag.ipynb                # Main training and experimentation notebook
├── Dockerfile                     # Containerized API runtime
├── requirements.txt               # Python dependencies
├── .gitattributes                 # Git LFS tracking rules for model files
└── .gitignore
```

## Dataset
Expected folder layout:
```text
../input/surface-crack-detection/
  Positive/
    *.jpg
  Negative/
    *.jpg
```

## Environment Setup
Use Python 3.10 for TensorFlow compatibility. On this machine, Python 3.10 is available at `/opt/homebrew/bin/python3.10`.

```bash
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
Run the notebook:
1. Open `risk-flag.ipynb`
2. Execute cells in order
3. The trained model is saved as `Model_Last_Prediction.h5`
4. Keep the saved model in the project root before running inference or the API

## Model Artifact
The repository is configured to store trained Keras model files with Git LFS:

```bash
git lfs install
git lfs pull
```

After cloning, verify that the real model file is present:

```bash
ls -lh Model_Last_Prediction.h5
```

The file should be hundreds of MB. If it is only a small text file, Git LFS has not downloaded the real model yet.

## Inference
```bash
python -m src.inference \
  --model Model_Last_Prediction.h5 \
  --image /absolute/path/to/image.jpg
```

Example output:
```text
label=Positive probability=0.8732
```

## API
Start the FastAPI app:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Predict from an uploaded image:

```bash
curl -X POST \
  -F "file=@/absolute/path/to/image.jpg" \
  http://localhost:8000/predict
```

The model path defaults to `Model_Last_Prediction.h5`. Override it with:

```bash
MODEL_PATH=/path/to/model.h5 uvicorn app:app --host 0.0.0.0 --port 8000
```

## Docker
Build and run the API:

```bash
docker build -t concrete-crack-api .
docker run --rm -p 8000:8000 concrete-crack-api
```

## Tests
Run lightweight smoke tests:

```bash
python -m unittest discover -s tests
```

## Notes
- Model input size: `200x200x3`
- Binary output with sigmoid activation
- Threshold defaults to `0.5` and can be changed with `--threshold`
