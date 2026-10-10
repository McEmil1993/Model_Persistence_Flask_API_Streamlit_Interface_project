from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

def main():
    # The sample dataset is for classroom practice only.
    data = pd.read_csv(DATA_PATH)
    X = data[["study_hours"]]
    y = data["passed"]

    model = LogisticRegression()
    model.fit(X, y)

    # Exercise 1: save the trained model under a new filename.
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH.name}")

    # Load the saved model and use it without retraining.
    loaded_model = joblib.load(MODEL_PATH)
    sample_hours = [[2], [5], [8]]
    predictions = loaded_model.predict(sample_hours)

    print("\nThree sample predictions:")
    for hours, prediction in zip([2, 5, 8], predictions):
        label = "Pass" if int(prediction) == 1 else "Needs Improvement"
        print(f"Study hours: {hours} -> {label} (class={int(prediction)})")

    print(
        "\nExplanation: the fitted model's learned parameters are stored in the "
        ".joblib file. Loading that file restores the trained model, so predictions "
        "can be made without fitting the model again. Retrain only when you want "
        "to learn from updated or different training data."
    )

if __name__ == "__main__":
    main()
