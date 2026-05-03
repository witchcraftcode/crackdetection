import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
import uvicorn

from model import predict

app = FastAPI(title="Concrete Crack Classification API")


@app.get("/")
def root():
    return {"message": "Concrete crack prediction API is running"}


@app.get("/health")
def health():
    model_path = os.getenv("MODEL_PATH", "Model_Last_Prediction.h5")
    return {"status": "ok", "model_path": model_path, "model_exists": os.path.exists(model_path)}


@app.post("/predict")
async def predict_risk(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        prediction = predict(temp_path)
        return {"prediction": prediction}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
