from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "student_pass_model.joblib"

def main():
    # Illustrative training data: study hours are the only model feature.
    study_hours = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    passed =     [0,   0,   0,   0,   0,   1,   1,   1,   1,   1,    1]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42))
    ])
    model.fit(study_hours, passed)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH}")

    loaded_model = joblib.load(MODEL_PATH)
    print("\nThree example predictions (loaded model):")
    for hours in [2, 5, 8]:
        prediction = int(loaded_model.predict([[hours]])[0])
        label = "Pass" if prediction == 1 else "Needs Improvement"
        print(f"Study hours: {hours:>2} -> {label}")

    print("\nThe model does not need retraining for these predictions because its learned parameters")
    print("were saved to a file and loaded again with joblib.load().")

if __name__ == "__main__":
    main()
