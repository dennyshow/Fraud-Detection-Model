from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import json

# Initialize FastAPI app
app = FastAPI(title="Multi-Model Fraud Detection API")

# Loading selected features used during training ===
selected_features_path = "model_output/selected_features.json"
if not os.path.exists(selected_features_path):
    raise FileNotFoundError("Missing selected_features.json")

with open(selected_features_path, "r") as f:
    selected_features = json.load(f)

# Loading all trained models from model_output ===
model_dir = "model_output"
models = {}

for file in os.listdir(model_dir):
    if file.endswith(".pkl"):
        model_name = file.replace(".pkl", "")
        try:
            models[model_name] = joblib.load(os.path.join(model_dir, file))
        except Exception as e:
            print(f"Error loading {file}: {e}")

# Confirm models are loading successfully
print(f"Loaded models: {list(models.keys())}")

# Define request schema for /predict endpoint ===
class Transaction(BaseModel):
    data: dict  # JSON payload with transaction features

# Using this for quick API health checks ===
@app.get("/")
def root():
    return {"message": "Fraud Detection API is running. Use /predict or /test-case?index=40"}

# Accepts client transaction input for prediction ===
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convert input dictionary to DataFrame and keep only selected features
        df = pd.DataFrame([transaction.data])[selected_features]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input features: {e}")

    predictions = {}

    # Loop through all models to generate predictions
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

# Run prediction on a known fraud sample from test set ===
@app.get("/test-case")
def predict_test_case(index: int = 40):
    try:
        # Load test data (already scaled)
        X_test = pd.read_csv("X_test_scaled.csv")

        # Get the row at the specified index and extract selected features
        row = X_test.loc[index][selected_features]
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Test case index {index} not valid: {e}")

    input_df = pd.DataFrame([row])
    results = {}

    # Loop through models and predict
    for name, model in models.items():
        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

            results[name] = {
                "prediction": int(pred),
                "fraud_probability": round(proba, 2) if proba is not None else "N/A"
            }

        except Exception as e:
            results[name] = {"error": str(e)}

    return {"results": results}
