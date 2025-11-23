# src/model/persistence.py
import os
from joblib import dump, load

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

def save_model(model, vectorizer):
    dump(model, MODEL_PATH)
    dump(vectorizer, VECTORIZER_PATH)

def load_model_and_vectorizer():
    model = load(MODEL_PATH)
    vectorizer = load(VECTORIZER_PATH)
    return model, vectorizer
