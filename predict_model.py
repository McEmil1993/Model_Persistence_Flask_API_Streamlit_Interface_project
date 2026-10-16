"""Exercise 1 - load the saved model and predict WITHOUT retraining."""
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "skilltrack_model.joblib"

model = joblib.load(MODEL_PATH)

for hours, attendance in [(2, 45), (5, 65), (8, 85)]:
    X = [[hours, attendance]]
    result = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    print(f"{hours} study hours, {attendance}% attendance -> {result} "
          f"(confidence: {prob:.0%})")