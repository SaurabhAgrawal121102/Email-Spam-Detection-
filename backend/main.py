from fastapi import FastAPI
from pydantic import BaseModel

from backend.predictor import predict

app = FastAPI(title="Spam Detection API")

class InputData(BaseModel):
    content: str

@app.get("/status")
def get_status():
    return {"status": "API is running"}

@app.post("/predict")
def predict_spam(data: InputData):
    input_data = data.model_dump()

    prediction = predict(input_data)

    return {"prediction": prediction}