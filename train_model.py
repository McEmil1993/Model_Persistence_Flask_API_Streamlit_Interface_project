from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

def main():
    # Example data: study hours and pass/fail label (1 = pass, 0 = fail).
    study_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float).reshape(-1, 1)
    passed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    model = LogisticRegression()
    model.fit(study_hours, passed)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    loaded_model = joblib.load(MODEL_PATH)
    for hours in [2, 5, 8]:
        result = int(loaded_model.predict([[hours]])[0])
        print(f"Study hours: {hours} -> Prediction: {'Pass' if result else 'Fail'}")
    print("The saved model can make predictions without retraining.")

if __name__ == "__main__":
    main()
