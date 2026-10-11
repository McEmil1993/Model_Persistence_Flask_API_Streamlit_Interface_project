from pathlib import Path

import joblib

from student_model import StudentPassModel


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def main() -> None:
    # This small example dataset is illustrative, not a real student record set.
    study_hours = [[hours] for hours in range(11)]
    passed = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    model = StudentPassModel().fit(study_hours, passed)

    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    loaded_model = joblib.load(MODEL_PATH)
    for hours in (2, 5, 8):
        prediction = loaded_model.predict([[hours]])[0]
        result = "Pass" if prediction == 1 else "Faigl"
        print(f"{hours} study hours: {result}")

    print(
        "The saved model contains the learned decision rule, so these predictions "
        "do not require training the model again."
    )


if __name__ == "__main__":
    main()
