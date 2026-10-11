"""Load the persisted classifier and produce validated predictions."""

import math
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from .paths import DEFAULT_MODEL_PATH, ProjectPath


def load_model(model_path: ProjectPath = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a locally generated model artifact."""
    artifact_path = Path(model_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {artifact_path}. Run 'python -m student_performance.train_model' first."
        )
    return joblib.load(artifact_path)


def predict_outcome(model: Pipeline, study_hours: float) -> str:
    """Validate study hours and return the trained model's outcome label."""
    if (
        isinstance(study_hours, bool)
        or not isinstance(study_hours, (int, float))
        or not math.isfinite(study_hours)
        or study_hours < 0
    ):
        raise ValueError("study_hours must be a finite, non-negative number.")

    return str(model.predict([[float(study_hours)]])[0])
