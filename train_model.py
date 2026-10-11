from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def train_and_save_model(model_path: Path = MODEL_PATH) -> LogisticRegression:
    study_hours = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    passed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    model = LogisticRegression()
    model.fit(study_hours, passed)
    joblib.dump(model, model_path)
    return model


def main() -> None:
    train_and_save_model()
    saved_model = joblib.load(MODEL_PATH)

    for hours in (2, 5, 8):
        prediction = int(saved_model.predict([[hours]])[0])
        result = "Pass" if prediction else "Not yet"
        print(f"{hours} study hours -> {result}")

    print(
        "The saved file contains the model's learned parameters. "
        "Loading it lets us predict without fitting the model again."
    )


if __name__ == "__main__":
    main()