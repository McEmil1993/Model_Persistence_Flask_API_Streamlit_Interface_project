"""
train_model.py

Exercise 1 - Model Persistence
Exercise 2 - trains the second model (study hours + attendance)

Run:  python train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "students.csv"

# Exercise 1: a different filename from the default model name.
HOURS_MODEL_FILE = BASE_DIR / "student_pass_model.joblib"
# Exercise 2: the model that also uses attendance.
FULL_MODEL_FILE = BASE_DIR / "student_pass_model_attendance.joblib"

LABELS = {0: "Fail", 1: "Pass"}
REQUIRED_COLUMNS = ["study_hours", "attendance_percentage", "passed"]


def load_data():
    """Read students.csv and stop with a clear message if it is not usable."""
    if not DATA_FILE.exists():
        raise SystemExit(f"{DATA_FILE.name} not found. Put it in the same folder as this script.")
    df = pd.read_csv(DATA_FILE)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{DATA_FILE.name} is missing column(s): {', '.join(missing)}")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if set(df["passed"].unique()) != {0, 1}:
        raise SystemExit("The 'passed' column must contain both 0 (fail) and 1 (pass).")
    return df


def train_and_save(df, features, filename):
    """Train a Logistic Regression model and save it with joblib.dump()."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["passed"], test_size=0.25, random_state=42, stratify=df["passed"]
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # The sample dataset is tiny, so this accuracy is only a rough indicator.
    accuracy = model.score(X_test, y_test)
    print(f"Trained on {features} -> test accuracy: {accuracy:.2f}")

    joblib.dump(model, filename)  # <-- save the trained model to a file
    print(f"Saved model to: {filename.name}")
    return model


def main():
    df = load_data()

    # --- Exercise 1: model that uses study hours only -----------------------
    print("=== Exercise 1: Model Persistence ===")
    train_and_save(df, ["study_hours"], HOURS_MODEL_FILE)

    # --- Exercise 2: model that uses study hours + attendance ---------------
    print("\n=== Exercise 2: model with attendance ===")
    train_and_save(df, ["study_hours", "attendance_percentage"], FULL_MODEL_FILE)

    # --- Load the saved model (no training happens here) --------------------
    print("\n=== Load the saved model and predict ===")
    loaded_model = joblib.load(HOURS_MODEL_FILE)  # <-- load it back from disk

    # The loaded model already holds the values it learned during training.
    print(
        f"Learned values inside the loaded file: "
        f"coefficient = {loaded_model.coef_[0][0]:.3f}, "
        f"intercept = {loaded_model.intercept_[0]:.3f}"
    )

    # Three predictions using different study hours.
    for hours in (1, 4, 8):
        sample = pd.DataFrame({"study_hours": [hours]})
        predicted_class = int(loaded_model.predict(sample)[0])
        pass_probability = loaded_model.predict_proba(sample)[0][1]
        print(
            f"Study hours = {hours:>2} -> {LABELS[predicted_class]} "
            f"(pass probability: {pass_probability:.0%})"
        )

    print(
        """
Why is no re-training needed?
  Training (.fit) is where the algorithm learns its values (the coefficients
  and intercept) from the data. Those learned values are stored inside the
  model object. joblib.dump() writes that whole object to a file, and
  joblib.load() restores it exactly as it was. So predict() only applies the
  already-learned values to the new input. It does not need .fit() or the
  training data again.
"""
    )


if __name__ == "__main__":
    main()
