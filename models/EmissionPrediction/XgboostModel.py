"""
XGBoost CO2 Emission Prediction Model
Dataset: EmissionDataset.csv
Target:  co2_g_per_km_new  (fuel-type-adjusted emission)
Features: speed_kmph, idle_pct, elevation_grad,
          vehicle_type (categorical), bs_norm (categorical), fuel_type (categorical)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import time

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("EmissionDataset.csv")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}\n")

# ─────────────────────────────────────────────
# 2. Basic EDA / Sanity Check
# ─────────────────────────────────────────────
print("=== Dataset Info ===")
print(df.describe(include="all"))
print(f"\nMissing values:\n{df.isnull().sum()}\n")

# ─────────────────────────────────────────────
# 3. Feature Engineering
# ─────────────────────────────────────────────
# Use the fuel-type-adjusted target
TARGET = "co2_g_per_km_new"

# All categorical columns including the new fuel_type
CATEGORICAL_COLS = ["vehicle_type", "bs_norm", "fuel_type"]

# Drop the old unadjusted target if it exists (keep only co2_g_per_km_new as target)
DROP_COLS = [TARGET, "co2_g_per_km"]  # drop both; we'll re-add TARGET as y

# Label-encode categorical columns
le_dict = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"  Encoded '{col}': {le.classes_}")

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df[TARGET]

print(f"\nFeatures used: {X.columns.tolist()}")
print(f"Target       : {TARGET}\n")

# ─────────────────────────────────────────────
# 4. Train / Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 5. XGBoost Model
# ─────────────────────────────────────────────
print("\nTraining XGBoost model...")
start = time.time()

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",        # fast histogram-based method
    random_state=42,
    n_jobs=-1,
    verbosity=1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

elapsed = time.time() - start
print(f"\nTraining finished in {elapsed:.1f}s")

# ─────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n=== Test-set Metrics ===")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

# ─────────────────────────────────────────────
# 7. Feature Importance
# ─────────────────────────────────────────────
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\n=== Feature Importance (gain) ===")
print(importance.sort_values(ascending=False))

# ─────────────────────────────────────────────
# 8. Save Model + Encoders + Feature Order
# ─────────────────────────────────────────────
MODEL_PATH = "xgboost_emission_model.pkl"
joblib.dump({
    "model": model,
    "label_encoders": le_dict,
    "feature_columns": X.columns.tolist(),   # preserve exact column order
}, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")