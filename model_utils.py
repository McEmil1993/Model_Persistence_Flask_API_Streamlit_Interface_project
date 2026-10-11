"""Functions shared by the Flask and Streamlit applications."""
from pathlib import Path
import joblib
import pandas as pd

SAVED_MODEL = Path(__file__).resolve().parent / "student_pass_model.joblib"


def get_classifier():
    if not SAVED_MODEL.is_file():
        raise FileNotFoundError("Run `python train_model.py` before starting the apps.")
    return joblib.load(SAVED_MODEL)


def get_prediction(hours, attendance):
    sample = pd.DataFrame({
        "study_hours": [float(hours)],
        "attendance_percentage": [float(attendance)],
    })
    model = get_classifier()
    predicted = int(model.predict(sample)[0])
    class_names = {1: "Pass", 0: "Needs improvement"}
    probabilities = model.predict_proba(sample)[0]
    pass_index = list(model.classes_).index(1)

    return {
        "prediction": predicted,
        "predicted_class": class_names[predicted],
        "pass_probability_percent": round(float(probabilities[pass_index]) * 100, 2),
    }
