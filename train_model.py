"""
Exercise 1 — Model Persistence
Train a student pass/fail model and save it with joblib.
Run: python train_model.py
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "students.csv"
MODEL_FILE = BASE_DIR / "student_pass_model.joblib"

def main():
    data = pd.read_csv(DATA_FILE)
    features = data[["study_hours"]]
    target = data["passed"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")

    # Save the trained model. You do not need to retrain to use this saved file.
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")

    # Load the saved model and make three sample predictions.
    loaded_model = joblib.load(MODEL_FILE)
    print("\nThree sample predictions:")
    for hours in [2, 5, 8]:
        sample = pd.DataFrame([{"study_hours": hours}])
        result = int(loaded_model.predict(sample)[0])
        label = "PASS" if result == 1 else "FAIL"
        print(f"Study hours: {hours} -> {label}")

if __name__ == "__main__":
    main()