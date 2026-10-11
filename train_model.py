from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def main():
    # Example training data: hours studied per day and whether the student passed.
    study_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]).reshape(-1, 1)
    passed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    model = LogisticRegression()
    model.fit(study_hours, passed)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    loaded_model = joblib.load(MODEL_PATH)
    for hours in (2, 6, 9):
        prediction = int(loaded_model.predict([[hours]])[0])
        label = "Pass" if prediction == 1 else "Fail"
        print(f"{hours} study hours/day -> {label} (class {prediction})")

    print(
        "Predictions use the learned parameters saved in the model file. "
        "Training is only needed again when the training data or model changes."
    )


if __name__ == "__main__":
    main()