## main.py
## API Server code for model serving

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the pre-trained model
model = joblib.load("model.pkl")

# Define the input schema
class Features(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "Model API is live!"}

@app.post("/predict")
def predict(data: Features):
    try:
        prediction = model.predict([data.features])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))