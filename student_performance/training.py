"""Train the student performance classifier and persist its pipeline."""

import csv
import math
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .paths import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, ProjectPath

VALID_OUTCOMES = {"Fail", "Pass"}


def train_model(
    data_path: ProjectPath = DEFAULT_DATA_PATH,
    model_path: ProjectPath = DEFAULT_MODEL_PATH,
) -> Path:
    """Fit a classifier from the CSV and persist the complete sklearn pipeline."""
    features: list[list[float]] = []
    outcomes: list[str] = []

    with Path(data_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"study_hours", "outcome"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Training data must contain study_hours and outcome columns.")

        for row_number, row in enumerate(reader, start=2):
            try:
                study_hours = float(row["study_hours"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid study_hours on CSV row {row_number}.") from error

            outcome = (row["outcome"] or "").strip()
            if not math.isfinite(study_hours) or study_hours < 0:
                raise ValueError(f"study_hours on CSV row {row_number} must be finite and non-negative.")
            if outcome not in VALID_OUTCOMES:
                raise ValueError(f"Invalid outcome on CSV row {row_number}; use Pass or Fail.")

            features.append([study_hours])
            outcomes.append(outcome)

    if not features or set(outcomes) != VALID_OUTCOMES:
        raise ValueError("Training data must include rows for both Pass and Fail outcomes.")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(features, outcomes)

    artifact_path = Path(model_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    return artifact_path
