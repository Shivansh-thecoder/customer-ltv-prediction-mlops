from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Customer LTV Prediction API"
)

# -------------------------
# Load Model Once
# -------------------------

model = joblib.load("models/rf_model.pkl")


# -------------------------
# Request Schema
# -------------------------

class CustomerData(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    AvgOrderValue: float
    TenureDays: float
    PurchaseRate: float
    RecencyRatio: float


# -------------------------
# Endpoint
# -------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    df = pd.DataFrame([customer.dict()])

    pred_log = model.predict(df)[0]

    pred_revenue = np.expm1(pred_log)

    return {
        "predicted_future_revenue": round(
            float(pred_revenue),
            2
        )
    }