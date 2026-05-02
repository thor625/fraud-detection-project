import pandas as pd
import uuid
from datetime import datetime

V_FEATURES = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES = V_FEATURES + ['Amount_scaled', 'Time_scaled']

def get_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def predict_single(model, transaction: dict) -> dict:
    df = pd.DataFrame([transaction])[ALL_FEATURES]
    prob = float(model.predict_proba(df)[:, 1][0])
    return {
        "transaction_id": str(uuid.uuid4()),
        "fraud_probability": round(prob, 4),
        "is_fraud": bool(prob >= 0.5),
        "risk_level": get_risk_level(prob),
        "timestamp": datetime.utcnow().isoformat()
    }

def predict_batch(model, df: pd.DataFrame) -> list:
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    probs = model.predict_proba(df[ALL_FEATURES])[:, 1]

    results = []
    for i, prob in enumerate(probs):
        results.append({
            "transaction_id": str(uuid.uuid4()),
            "row": i,
            "fraud_probability": round(float(prob), 4),
            "is_fraud": bool(prob >= 0.5),
            "risk_level": get_risk_level(prob),
            "timestamp": datetime.utcnow().isoformat()
        })

    return results