from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ✅ Define the FastAPI app before using it
app = FastAPI()

# ✅ Load your trained model
model = joblib.load("model.pkl")

# ✅ Define a request body model
class InputData(BaseModel):
    features: list

@app.get("/")
def read_root():
    return {"message": "ML Model API is up!"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}