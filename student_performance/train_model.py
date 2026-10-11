"""Train and persist the student performance classifier."""

from .model import load_model, predict_outcome
from .paths import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH
from .training import train_model


def main() -> None:
    model_path = train_model(DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH)
    print(f"Saved trained model to {model_path}")

    saved_model = load_model(model_path)
    for study_hours in (2, 5, 8):
        outcome = predict_outcome(saved_model, study_hours)
        print(f"Prediction for {study_hours} study hours: {outcome}")


if __name__ == "__main__":
    main()
