"""
Realistic Student Pass Prediction Model
---------------------------------------
- Features: [study_hours, attendance_percentage]  (INDEPENDENT)
- 200 synthetic samples with realistic noise
- Pipeline: StandardScaler + LogisticRegression
- Saves: model + metadata for reproducibility
"""

import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# 2. GENERATE REALISTIC SYNTHETIC DATA
# ============================================================
N = 200

# Independent features
study_hours = np.random.uniform(0, 15, N)       # 0 to 15 hours/week
attendance  = np.random.uniform(30, 100, N)     # 30% to 100%

# Real-world logic:
#   - Passing is driven by BOTH study and attendance
#   - But with noise (some students pass with low effort, some fail with high effort)
#   - There's a fuzzy threshold, not a clean line
score = (
    0.35 * study_hours            # study is the bigger factor
    + 0.045 * attendance          # attendance matters but less
    + np.random.normal(0, 1.5, N) # random noise / life events
)
y = (score > 5.5).astype(int)

# Build the feature matrix in the EXACT order the model expects
X = np.column_stack([study_hours, attendance])

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total samples       : {N}")
print(f"Pass (1) count      : {y.sum()}")
print(f"Fail (0) count      : {N - y.sum()}")
print(f"Pass rate           : {y.mean():.2%}")
print(f"Study hours range   : {study_hours.min():.2f} - {study_hours.max():.2f}")
print(f"Attendance range    : {attendance.min():.2f} - {attendance.max():.2f}")
print()

# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

# ============================================================
# 4. BUILD PIPELINE (Scaler + Logistic Regression)
# ============================================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver="lbfgs",
    )),
])

# ============================================================
# 5. TRAIN
# ============================================================
pipeline.fit(X_train, y_train)
print("Model trained on", len(X_train), "samples.")

# ============================================================
# 6. EVALUATE
# ============================================================
y_pred = pipeline.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

print()
print("=" * 60)
print("EVALUATION ON TEST SET (20% held out)")
print("=" * 60)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"], zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# 7. INSPECT LEARNED COEFFICIENTS
# ============================================================
model_step = pipeline.named_steps["model"]
feature_names = ["study_hours", "attendance_percentage"]

print()
print("=" * 60)
print("LEARNED COEFFICIENTS (on scaled features)")
print("=" * 60)
for name, coef in zip(feature_names, model_step.coef_[0]):
    print(f"  {name:25s}: {coef:+.4f}")
print(f"  {'intercept':25s}: {model_step.intercept_[0]:+.4f}")
print()
print("Interpretation: larger absolute coefficient = stronger influence.")
print("(Positive → pushes toward Pass; negative → pushes toward Fail.)")

# ============================================================
# 8. SAVE MODEL + METADATA
# ============================================================
model_path = "student_pass_model.joblib"
joblib.dump(pipeline, model_path)
print()
print(f"✅ Model saved -> {model_path}")

metadata = {
    "model_type": "LogisticRegression (with StandardScaler)",
    "feature_order": feature_names,
    "n_features": len(feature_names),
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "random_state": RANDOM_STATE,
    "trained_at": datetime.now().isoformat(timespec="seconds"),
    "metrics": {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    },
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved -> model_metadata.json")

# ============================================================
# 9. SANITY CHECK — reload and predict
# ============================================================
print()
print("=" * 60)
print("SANITY CHECK (reload from disk)")
print("=" * 60)
loaded = joblib.load(model_path)
test_cases = [
    (1,  40, "low study, low attendance"),
    (14, 40, "high study, low attendance"),
    (2,  95, "low study, high attendance"),
    (7,  80, "moderate study, decent attendance"),
    (14, 95, "high study, high attendance"),
]
for h, a, desc in test_cases:
    pred = loaded.predict([[h, a]])[0]
    prob = loaded.predict_proba([[h, a]])[0][1]
    label = "Pass" if pred == 1 else "Fail"
    print(f"  study={h:>2}h  att={a:>3}%  -> {label:4s}  (P(pass)={prob:.2f})   [{desc}]")