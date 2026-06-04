import joblib
import pandas as pd
import numpy as np

# -------------------------
# Load Model
# -------------------------

model = joblib.load("models/rf_model.pkl")

# -------------------------
# Sample Customer
# -------------------------

sample = pd.DataFrame([{
    "Recency": 20,
    "Frequency": 5,
    "Monetary": 2000,
    "AvgOrderValue": 400,
    "TenureDays": 300,
    "PurchaseRate": 5 / 300,
    "RecencyRatio": 20 / 300
}])

# -------------------------
# Predict
# -------------------------

pred_log = model.predict(sample)[0]

# -------------------------
# Convert Back
# -------------------------

pred_revenue = np.expm1(pred_log)

print(f"Predicted Future Revenue: {pred_revenue:.2f}")