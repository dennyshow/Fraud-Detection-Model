from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import json

app = FastAPI(title="Multi-Model Fraud Detection API")

# Load selected features
selected_features_path = "model_output/selected_features.json"
if not os.path.exists(selected_features_path):
    raise FileNotFoundError("Missing selected_features.json")

with open(selected_features_path, "r") as f:
    selected_features = json.load(f)

# Load all models in model_output folder
model_dir = "model_output"
models = {}

for file in os.listdir(model_dir):
    if file.endswith(".pkl"):
        model_name = file.replace(".pkl", "")
        try:
            models[model_name] = joblib.load(os.path.join(model_dir, file))
        except Exception as e:
            print(f"Error loading {file}: {e}")

print(f"Loaded models: {list(models.keys())}")

# Define the request schema
class Transaction(BaseModel):
    data: dict

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Build input DataFrame with selected features
        df = pd.DataFrame([transaction.data])[selected_features]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input features: {e}")

    predictions = {}

    for name, model in models.items():
        try:
            pred = model.predict(df)[0]
            proba = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else None
            predictions[name] = {
                "prediction": int(pred),
                "fraud_probability": round(proba, 2) if proba is not None else "N/A"
            }
        except Exception as e:
            predictions[name] = {"error": str(e)}

    return {"results": predictions}
