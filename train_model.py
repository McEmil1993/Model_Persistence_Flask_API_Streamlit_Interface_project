from pathlib import Path

import joblib

from student_model import StudentPassModel


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def main() -> None:
    study_hours = [[hours] for hours in range(11)]
    passed = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    model = StudentPassModel()
    model.fit(study_hours, passed)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH.name}")

    saved_model = joblib.load(MODEL_PATH)
    for hours in (2, 5, 8):
        prediction = int(saved_model.predict([[hours]])[0])
        result = "Pass" if prediction == 1 else "Fail"
        print(f"{hours} study hours: {result} (class {prediction})")


if __name__ == "__main__":
    main()