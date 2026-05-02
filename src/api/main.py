from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, create_model
import pandas as pd
import io
import os
from dotenv import load_dotenv
from pathlib import Path
from src.lib.connectors import Connectors
from src.api.model import predict_single, predict_batch, ALL_FEATURES, V_FEATURES

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / '.env')

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time and batch credit card fraud detection powered by XGBoost",
    version="1.0.0"
)

# ── Startup ─────────────────────────────────
s3 = Connectors.connect_s3()
model = Connectors.load_model(s3)
dynamodb = Connectors.get_dynamodb()

# ── Request model ────────────────────────────
field_definitions = {
    'Time_scaled': (float, ...),
    'Amount_scaled': (float, ...),
}
for v in V_FEATURES:
    field_definitions[v] = (float, ...)

Transaction = create_model('Transaction', **field_definitions)

# ── Routes ───────────────────────────────────
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
    from datetime import datetime
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        result = predict_single(model, transaction.model_dump())
        Connectors.save_prediction(dynamodb, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        results = predict_batch(model, df)

        for result in results:
            Connectors.save_prediction(dynamodb, result)

        fraud_count = sum(1 for r in results if r['is_fraud'])

        return {
            "filename": file.filename,
            "total_transactions": len(results),
            "fraud_detected": fraud_count,
            "fraud_rate": round(fraud_count / len(results) * 100, 3),
            "results": results
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))