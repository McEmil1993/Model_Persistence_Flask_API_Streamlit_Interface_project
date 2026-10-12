from pathlib import Path

import joblib

from student_model import EXAMPLE_PASS_LABELS, EXAMPLE_STUDY_HOURS, StudentPassModel


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def main() -> None:
    model = StudentPassModel()
    model.fit(EXAMPLE_STUDY_HOURS, EXAMPLE_PASS_LABELS)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH.name}")

    saved_model = joblib.load(MODEL_PATH)
    for hours in (2, 5, 8):
        prediction = int(saved_model.predict([[hours]])[0])
        result = "Pass" if prediction == 1 else "Fail"
        print(f"{hours} study hours: {result} (class {prediction})")


if __name__ == "__main__":
    main()