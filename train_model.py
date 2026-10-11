"""
Jay Mark Renos - Student Pass Prediction
Train the model once, save it, then reuse it from Flask or Streamlit.
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"


def main():
    # Example training data: study hours and exam outcome (0 = Fail, 1 = Pass).
    # Replace this sample data with your teacher's dataset if one was provided.
    study_hours = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5,
                            5, 5.5, 6, 6.5, 7, 7.5, 8]).reshape(-1, 1)
    passed = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    model = LogisticRegression(random_state=42)
    model.fit(study_hours, passed)

    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH.name}")

    # Demonstrate loading the saved model and making three predictions.
    loaded_model = joblib.load(MODEL_PATH)
    for hours in [2, 4, 6]:
        prediction = int(loaded_model.predict([[hours]])[0])
        label = "Pass" if prediction == 1 else "Fail"
        probability = float(loaded_model.predict_proba([[hours]])[0][1])
        print(f"Study hours: {hours} -> {label} (pass probability: {probability:.1%})")

    print("\nThe saved model can be loaded again, so training is not required for each prediction.")


if __name__ == "__main__":
    main()
