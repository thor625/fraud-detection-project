from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import boto3
import io
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / '.env')

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time and batch credit card fraud detection powered by XGBoost",
    version="1.0.0"
)

# Load model from S3 on startup
def load_model():
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )
    obj = s3.get_object(
        Bucket=os.getenv('S3_BUCKET'),
        Key='model-artifacts/xgboost_model.pkl'
    )
    return pickle.loads(obj['Body'].read())

model = load_model()

# DynamoDB client
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

# ── Models ──────────────────────────────────────────────
class Transaction(BaseModel):
    Time_scaled: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount_scaled: float

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    timestamp: str

# ── Helper ───────────────────────────────────────────────
def get_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def save_prediction(prediction: dict, table_name: str = "fraud-predictions"):
    try:
        table = dynamodb.Table(table_name)
        table.put_item(Item=prediction)
    except Exception as e:
        print(f"DynamoDB write failed: {e}")

# ── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/predict", "/upload", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        # Convert to dataframe
        data = pd.DataFrame([transaction.dict()])
        
        # Score
        prob = float(model.predict_proba(data)[:, 1][0])
        is_fraud = prob >= 0.5
        
        # Build result
        result = {
            "transaction_id": str(uuid.uuid4()),
            "fraud_probability": round(prob, 4),
            "is_fraud": is_fraud,
            "risk_level": get_risk_level(prob),
            "timestamp": datetime.utcnow().isoformat()
        }

        # Log to DynamoDB
        save_prediction(result)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required_cols = [c for c in Transaction.__fields__.keys()]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        # Score all rows
        probs = model.predict_proba(df[required_cols])[:, 1]
        
        results = []
        for i, prob in enumerate(probs):
            result = {
                "transaction_id": str(uuid.uuid4()),
                "row": i,
                "fraud_probability": round(float(prob), 4),
                "is_fraud": bool(prob >= 0.5),
                "risk_level": get_risk_level(prob),
                "timestamp": datetime.utcnow().isoformat()
            }
            results.append(result)
            save_prediction(result)

        fraud_count = sum(1 for r in results if r['is_fraud'])

        return {
            "filename": file.filename,
            "total_transactions": len(results),
            "fraud_detected": fraud_count,
            "fraud_rate": round(fraud_count / len(results) * 100, 3),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))