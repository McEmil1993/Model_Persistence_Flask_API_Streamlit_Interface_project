"""
Exercise 1 — Model Persistence
Train a simple model to predict whether a student passes based on study hours,
then save it to student_pass_model.joblib.
"""
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

# Example training data for practice only: [study_hours] -> pass (1) / fail (0).
# Replace this sample data with your instructor's dataset if one was provided.
X = [[0.5], [1], [1.5], [2], [2.5], [3], [3.5], [4], [4.5], [5], [6], [7]]
y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

def main():
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH.name}")

    # Load the saved model back from disk.
    loaded_model = joblib.load(MODEL_PATH)
    print("\nThree example predictions:")
    for hours in [1.0, 3.0, 5.0]:
        prediction = int(loaded_model.predict([[hours]])[0])
        label = "Pass" if prediction == 1 else "Fail"
        print(f"{hours:.1f} study hour(s) -> {label} (class {prediction})")

if __name__ == "__main__":
    main()
