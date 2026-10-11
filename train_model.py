"""Train, save, load, and use a simple student pass-prediction model."""

from math import exp
from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def train_and_save_model() -> dict[str, float]:
    """Train a model where 1 means pass and 0 means fail, then save it."""
    # Each row contains one feature: study hours.  Labels: 0 = Fail, 1 = Pass.
    study_hours = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    passed = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    failed_hours = [hours[0] for hours, label in zip(study_hours, passed) if label == 0]
    passed_hours = [hours[0] for hours, label in zip(study_hours, passed) if label == 1]
    model = {"pass_threshold": (max(failed_hours) + min(passed_hours)) / 2}
    joblib.dump(model, MODEL_PATH)
    return model


def load_model() -> dict[str, float]:
    """Load the persisted model; create it first when it is not present."""
    if not MODEL_PATH.exists():
        train_and_save_model()
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        # Recreate a model saved by an earlier incompatible implementation.
        model = train_and_save_model()
    if not isinstance(model, dict) or "pass_threshold" not in model:
        model = train_and_save_model()
    return model


def _adjusted_hours(model: dict[str, float], hours: float, attendance_percentage: float = 75.0) -> float:
    """Combine study hours with the attendance adjustment stored in the model."""
    reference = model.get("attendance_reference", 75.0)
    weight = model.get("attendance_weight", 1.0)
    return hours + ((attendance_percentage - reference) / 25.0) * weight


def prediction_label(
    model: dict[str, float], hours: float, attendance_percentage: float = 75.0
) -> str:
    """Return a readable pass/fail label using study hours and attendance."""
    return "Pass" if _adjusted_hours(model, hours, attendance_percentage) >= model["pass_threshold"] else "Fail"


def pass_probability(model: dict[str, float], hours: float, attendance_percentage: float = 75.0) -> float:
    """Return an estimated probability of passing, from 0 through 1."""
    return 1 / (1 + exp(-(_adjusted_hours(model, hours, attendance_percentage) - model["pass_threshold"])))


if __name__ == "__main__":
    train_and_save_model()
    saved_model = load_model()

    print(f"Model saved to: {MODEL_PATH.name}")
    for hours in (2, 5, 8):
        print(f"{hours} study hours: {prediction_label(saved_model, hours)}")

    print(
        "\nThe model is loaded from student_pass_model.joblib, so it can make "
        "predictions without being trained again."
    )
