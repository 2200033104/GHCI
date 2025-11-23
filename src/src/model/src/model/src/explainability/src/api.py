# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import os
from src.model.persistence import load_model_and_vectorizer
from src.model.predict import predict_single
from src.preprocessing import lemmatize, extract_signals
import json

app = FastAPI(title="IntelliClassify Transaction Categorisation API")

# Load model at startup
MODEL, VECT = None, None

class PredictRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

@app.on_event("startup")
def startup_event():
    global MODEL, VECT
    try:
        MODEL, VECT = load_model_and_vectorizer()
        print("Model & vectorizer loaded on startup.")
    except Exception as e:
        print("Warning: Could not load model on startup:", e)
        MODEL, VECT = None, None

@app.get("/")
def root():
    return {"status": "ok", "service": "IntelliClassify"}

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text
    processed = lemmatize(text)
    if MODEL is None:
        return {"error": "model not available", "processed_text": processed}
    cat, conf = predict_single(processed, MODEL)
    signals = extract_signals(text)
    return {"raw_text": text, "processed_text": processed, "predicted_category": cat, "confidence": conf, "signals": signals}

@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    results = []
    for t in req.texts:
        processed = lemmatize(t)
        if MODEL:
            cat, conf = predict_single(processed, MODEL)
        else:
            cat, conf = "unknown", 0.0
        results.append({"raw_text": t, "processed_text": processed, "predicted_category": cat, "confidence": conf})
    return {"results": results}

if __name__ == "__main__":
    # recommended: run with uvicorn src.api:app --reload
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
