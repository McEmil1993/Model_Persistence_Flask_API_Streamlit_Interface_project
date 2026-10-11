"""Train and persist a demo student pass/fail classifier."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_pass_model.joblib"

def main():
    # Illustrative teaching data only; replace with your real, approved dataset.
    study_hours = np.array([0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9]).reshape(-1, 1)
    outcomes = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    model.fit(study_hours, outcomes)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
