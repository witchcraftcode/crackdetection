# Concrete Surface Crack Classification (CNN)

CNN-based binary classifier for concrete surface crack detection.

## Overview
This project predicts whether a concrete surface image contains a crack:
- `Positive`: crack present
- `Negative`: no visible crack

The current training workflow is in the notebook `risk-flag.ipynb`, and reusable code is available in `src/`.

## Project Structure
- `risk-flag.ipynb` - main training and experimentation notebook
- `src/model.py` - reusable CNN model builder
- `src/inference.py` - CLI inference for single-image prediction
- `requirements.txt` - Python dependencies

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
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
Run the notebook:
1. Open `risk-flag.ipynb`
2. Execute cells in order
3. The trained model is saved as `Model_Last_Prediction.h5`

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

## Notes
- Model input size: `200x200x3`
- Binary output with sigmoid activation
- Threshold defaults to `0.5` and can be changed with `--threshold`
