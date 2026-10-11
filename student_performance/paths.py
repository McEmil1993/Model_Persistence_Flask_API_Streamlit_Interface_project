"""Shared project paths for training and inference."""

from pathlib import Path
from typing import TypeAlias

ProjectPath: TypeAlias = str | Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "student_pass_model.joblib"
