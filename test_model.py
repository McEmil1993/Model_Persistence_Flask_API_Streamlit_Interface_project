from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The saved model was not found. Run train_model.py first."
        )

    model = joblib.load(MODEL_PATH)
    study_hours_examples = [2, 5, 8]

    print("Predictions from the saved model:")
    for study_hours in study_hours_examples:
        predicted_class = int(model.predict([[study_hours]])[0])
        result = "Pass" if predicted_class == 1 else "Fail"
        print(f"- {study_hours} study hours: {result}")

    print(
        "\nThe model was loaded from student_pass_model.joblib, so it did not "
        "need to be trained again."
    )


if __name__ == "__main__":
    main()
