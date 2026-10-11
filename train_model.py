from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def main() -> None:
    training_features = [[1], [2], [3], [4], [5], [6], [7], [8]]

    training_labels = [0, 0, 0, 0, 1, 1, 1, 1]

    model = LogisticRegression()
    model.fit(training_features, training_labels)

    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
