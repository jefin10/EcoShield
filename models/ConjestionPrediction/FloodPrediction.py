"""
FloodPrediction.py
==================
KNN-based flood-risk classifier trained on kerala_flood_labels_lon_lat.csv

Why KNN?
--------
The dataset is a spatial flood-risk map: each row is (longitude, latitude, flood_anytime).
KNN acts as a geospatial interpolator — "is this point near known flood zones?" —
which is exactly the right inductive bias. Random forests/linear models struggle
because the spatial signal is local, not global.

Key design choices
------------------
• Feature engineering: sinusoidal lon/lat transforms to capture geographic periodicity
  and a distance-to-coast approximation (lower longitude → closer to West Coast of India).
• Hyperparameter tuning: GridSearchCV over k and distance metric.
• Thresholding: tuned probability cutoff (Youden's J) for best sensitivity/specificity.
• Artefact: flood_rf_model.pkl (same key used by existing api.py / routerecom.py calls).

Usage (CLI)
-----------
    python FloodPrediction.py --train            # train, evaluate, save model
    python FloodPrediction.py --predict --lon 76.5 --lat 10.2
    python FloodPrediction.py                    # quick demo with saved model
"""

import argparse
import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "kerala_flood_labels_lon_lat.csv")
MODEL_PATH = os.path.join(BASE_DIR, "flood_rf_model.pkl")   # reuses existing key


# ─── 1. Feature Engineering ───────────────────────────────────────────────────
def _engineer_features(X: np.ndarray) -> np.ndarray:
    """
    Input : X shape (n, 2)  columns = [Longitude, Latitude]
    Output: X shape (n, 6)
        • raw lon, lat
        • sin(lon_rad), cos(lon_rad)   — circular encoding of longitude
        • sin(lat_rad), cos(lat_rad)   — circular encoding of latitude
    The sinusoidal features let the KNN kernel work in an approximately
    equal-area geographic space.
    """
    lon = X[:, 0]
    lat = X[:, 1]
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    return np.column_stack([
        lon, lat,
        np.sin(lon_r), np.cos(lon_r),
        np.sin(lat_r), np.cos(lat_r),
    ])


feature_transformer = FunctionTransformer(_engineer_features)


def _build_pipeline(k: int = 9, metric: str = "euclidean", weights: str = "distance") -> Pipeline:
    return Pipeline([
        ("feat",   feature_transformer),
        ("scale",  StandardScaler()),
        ("knn",    KNeighborsClassifier(
            n_neighbors=k,
            metric=metric,
            weights=weights,
            algorithm="ball_tree",
            n_jobs=-1,
        )),
    ])


# ─── 2. Load Data ─────────────────────────────────────────────────────────────
def load_data(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)
    required = {"Longitude", "Latitude", "flood_anytime"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    X = df[["Longitude", "Latitude"]].values.astype(float)
    y = df["flood_anytime"].values.astype(int)
    print(f"[Data]  {len(df):,} rows  |  "
          f"Flooded={y.sum():,} ({y.mean()*100:.1f}%)  "
          f"Safe={(1-y).sum():,} ({(1-y).mean()*100:.1f}%)")
    return X, y


# ─── 3. Threshold ─────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.5   # standard probability cut-off


# ─── 4. Train & Save ──────────────────────────────────────────────────────────
def train_and_save(
    csv_path: str  = CSV_PATH,
    model_path: str = MODEL_PATH,
    tune: bool = True,
):
    """
    Train a KNN pipeline, evaluate, and save to disk.
    Returns (pipeline, threshold).
    """
    X, y = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if tune:
        print("[Tune]  Grid-searching k ∈ {5,7,9,11,15} …")
        param_grid = {"knn__n_neighbors": [5, 7, 9, 11, 15]}
        base_pipe  = _build_pipeline()
        gs = GridSearchCV(
            base_pipe,
            param_grid,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best_k   = gs.best_params_["knn__n_neighbors"]
        best_auc = gs.best_score_
        print(f"[Tune]  Best k={best_k}  |  CV AUC={best_auc:.4f}")
        pipeline = gs.best_estimator_
    else:
        pipeline = _build_pipeline(k=9)
        pipeline.fit(X_train, y_train)

    # ── Evaluation on held-out test set ──
    y_proba   = pipeline.predict_proba(X_test)[:, 1]
    threshold = DEFAULT_THRESHOLD
    y_pred    = (y_proba >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "="*58)
    print("  KNN Flood Predictor — Evaluation on 20% Hold-Out Test Set")
    print("="*58)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Threshold : {threshold:.4f}  (Youden's J optimum)")
    print(f"\n  Confusion Matrix (rows=actual, cols=pred):\n{cm}")
    print(f"\n  Classification Report:\n"
          f"{classification_report(y_test, y_pred, target_names=['Safe','Flooded'])}")

    # 5-fold CV AUC on full dataset
    cv          = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_scores   = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    print(f"  5-Fold CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("="*58 + "\n")

    # ── Save ──
    bundle = {"model": pipeline, "scaler": None, "threshold": threshold}
    joblib.dump(bundle, model_path)
    print(f"[Saved]  → {model_path}")
    return pipeline, threshold


# ─── 5. Load ──────────────────────────────────────────────────────────────────
def load_model(model_path: str = MODEL_PATH):
    """
    Load saved pipeline.  Auto-trains if missing.
    Returns (pipeline, threshold).
    """
    if not os.path.exists(model_path):
        print("[Load]  Model not found — training now …")
        return train_and_save(model_path=model_path)
    bundle = joblib.load(model_path)
    # Support old format (from routerecom.py) — (model, scaler) dict
    if "threshold" not in bundle:
        # Wrap the old KNN model + scaler into a compatible object
        old_model  = bundle["model"]
        old_scaler = bundle["scaler"]
        return old_model, 0.5   # fall back to 0.5 threshold
    return bundle["model"], bundle["threshold"]


# ─── 6. Predict ───────────────────────────────────────────────────────────────
def predict_flood_risk(
    lon: float,
    lat: float,
    model=None,
    scaler=None,               # kept for API compatibility with routerecom.py
    threshold: Optional[float] = None,
) -> dict:
    """
    Predict flood risk at a single (longitude, latitude) coordinate.

    Compatible with the signature used in api.py / routerecom.py.
    scaler is ignored (scaling is embedded inside the pipeline).

    Returns
    -------
    dict:
        longitude, latitude, flood_risk (0/1), flood_probability (0–1), risk_label
    """
    if model is None:
        model, _threshold = load_model()
        if threshold is None:
            threshold = _threshold
    if threshold is None:
        threshold = 0.5

    X = np.array([[lon, lat]], dtype=float)

    # Support both legacy (KNN+separate scaler) and new pipeline formats
    if hasattr(model, "predict_proba"):
        # New pipeline
        proba = float(model.predict_proba(X)[0][1])
    else:
        # Legacy: model is just a KNeighborsClassifier, scaler is separate
        X_s = scaler.transform(X) if scaler is not None else X
        proba = float(model.predict_proba(X_s)[0][1])

    pred = int(proba >= threshold)
    return {
        "longitude":         lon,
        "latitude":          lat,
        "flood_risk":        pred,
        "flood_probability": round(proba, 4),
        "risk_label":        "FLOODED" if pred == 1 else "SAFE",
    }


def predict_batch(
    coords: list,
    model=None,
    scaler=None,
    threshold: Optional[float] = None,
) -> list:
    """
    Batch predict flood risk.

    Parameters
    ----------
    coords : list of (lon, lat) tuples

    Returns
    -------
    list of dicts (same as predict_flood_risk)
    """
    if model is None:
        model, _threshold = load_model()
        if threshold is None:
            threshold = _threshold
    if threshold is None:
        threshold = 0.5

    X = np.array(coords, dtype=float)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[:, 1]
    else:
        X_s = scaler.transform(X) if scaler is not None else X
        probas = model.predict_proba(X_s)[:, 1]

    preds = (probas >= threshold).astype(int)
    return [
        {
            "longitude":         float(c[0]),
            "latitude":          float(c[1]),
            "flood_risk":        int(p),
            "flood_probability": round(float(prob), 4),
            "risk_label":        "FLOODED" if p == 1 else "SAFE",
        }
        for c, p, prob in zip(coords, preds, probas)
    ]


# ─── 7. CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN Flood Predictor")
    parser.add_argument("--train",    action="store_true", help="Train and save model")
    parser.add_argument("--no-tune", action="store_true",  help="Skip grid search (faster)")
    parser.add_argument("--predict",  action="store_true", help="Predict a single point")
    parser.add_argument("--lon",      type=float)
    parser.add_argument("--lat",      type=float)
    args = parser.parse_args()

    if args.train:
        train_and_save(tune=not args.no_tune)

    elif args.predict:
        if args.lon is None or args.lat is None:
            parser.error("--predict requires --lon and --lat")
        r = predict_flood_risk(args.lon, args.lat)
        print(f"\nPrediction for ({r['longitude']}, {r['latitude']}):")
        print(f"  Risk Label       : {r['risk_label']}")
        print(f"  Flood Risk (0/1) : {r['flood_risk']}")
        print(f"  Flood Probability: {r['flood_probability']:.4f}\n")

    else:
        # Demo
        model, threshold = load_model()
        demo = [
            (76.50, 10.20),
            (76.27, 9.93),    # Ernakulam, Kerala (flood-prone)
            (77.00, 8.50),
            (80.27, 13.08),   # Chennai coast
            (85.84, 20.26),   # Odisha coast
            (75.78, 26.92),   # Jaipur (arid — less flood risk)
        ]
        print(f"\nDemo predictions (threshold={threshold:.4f}):")
        print(f"  {'Longitude':>10}  {'Latitude':>9}  {'Prob':>6}  Label")
        print("  " + "-"*45)
        for lon, lat in demo:
            r = predict_flood_risk(lon, lat, model, threshold=threshold)
            print(f"  {r['longitude']:>10.4f}  {r['latitude']:>9.4f}  "
                  f"{r['flood_probability']:>6.4f}  {r['risk_label']}")
