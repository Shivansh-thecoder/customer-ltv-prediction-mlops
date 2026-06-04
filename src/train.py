import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/model_dataset.csv")

# -------------------------
# Features & Target
# -------------------------

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "TenureDays",
    "PurchaseRate",
    "RecencyRatio"
]

X = df[features]

y = df["FutureRevenue_Log"]

# -------------------------
# Split Data
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Train Model
# -------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------
# Predict
# -------------------------

y_pred = model.predict(X_test)

# -------------------------
# Metrics
# -------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# -------------------------
# Save Model
# -------------------------

joblib.dump(
    model,
    "models/rf_model.pkl"
)

print("Model saved successfully.")