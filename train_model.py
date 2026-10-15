
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# --- Configuration -----------------------------------------------------------

# Anchor all file paths to this script's folder, not the current working
# directory, so the script behaves the same no matter where it is run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILENAME = "student_data.csv"
MODEL_FILENAME = "student_pass_model.joblib"

DATA_PATH = os.path.join(BASE_DIR, DATA_FILENAME)
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# Fixed seed: every run produces the same dataset, so results are reproducible.
RANDOM_SEED = 42
N_STUDENTS = 100

# Study load is now measured per DAY: a student can realistically study 0-8
# hours in a day, and the pass threshold sits near 3 hours/day.
HOURS_RANGE = (0.0, 8.0)
PASS_THRESHOLD_HOURS = 3.0

# The three study-hour values used for the demonstration predictions.
PREDICTION_HOURS = [1.0, 3.0, 6.0]


def generate_dataset(n_students=N_STUDENTS, seed=RANDOM_SEED):
    """Build a synthetic student dataset and write it to CSV.

    Study load is measured per day: hours range from 0 to 8, and the pass
    probability crosses 0.5 near 3 hours/day. Each student's outcome is sampled
    from that probability with a weighted coin, which gives the data realistic
    noise instead of a perfectly separable split.

    attendance_percentage is generated and stored even though the model below
    ignores it. It is included so the data is already available if you later
    retrain the model to use attendance as a second input.
    """
    rng = np.random.default_rng(seed)

    lower, upper = HOURS_RANGE
    study_hours = np.round(rng.uniform(lower, upper, n_students), 1)
    attendance_percentage = np.round(rng.uniform(40, 100, n_students), 1)

    # Logistic curve centred on PASS_THRESHOLD_HOURS; steepness is fixed at 1.2.
    pass_probability = 1 / (1 + np.exp(-1.2 * (study_hours - PASS_THRESHOLD_HOURS)))
    passed = (rng.uniform(0, 1, n_students) < pass_probability).astype(int)

    dataset = pd.DataFrame(
        {
            "student_id": np.arange(1, n_students + 1),
            "study_hours": study_hours,
            "attendance_percentage": attendance_percentage,
            "passed": passed,
        }
    )
    dataset.to_csv(DATA_PATH, index=False)

    print(f"Generated {DATA_FILENAME} with {n_students} students.")
    print(f"  Rows where the student passed: {int(passed.sum())} / {n_students}")
    print("  Attendance column is present but unused by the model (see note below).")
    return dataset


def train_and_save(dataset):
    """Train logistic regression on study hours and save it with joblib.dump()."""
    # The model uses study_hours ONLY. Pandas is given a 2-D frame because
    # scikit-learn expects features shaped (n_samples, n_features).
    features = dataset[["study_hours"]]
    target = dataset["passed"]

    model = LogisticRegression()
    model.fit(features, target)

    joblib.dump(model, MODEL_PATH)
    print(f"\nTrained model saved to {MODEL_FILENAME} using joblib.dump().")
    print(f"  Learned coefficient: {model.coef_[0][0]:.4f}")
    print(f"  Learned intercept:   {model.intercept_[0]:.4f}")
    return model


def load_and_predict(hours_list=PREDICTION_HOURS):
    """Load the model from disk and predict for several study-hour values.

    joblib.load() rebuilds the trained model exactly as it was saved, including
    its learned coefficients. This is why no retraining is needed — see the
    explanation printed at the end of main().
    """
    model = joblib.load(MODEL_PATH)
    print(f"\nLoaded model from {MODEL_FILENAME} using joblib.load().")

    print("\nMaking three predictions:")
    for hours in hours_list:
        # One row, one column — the same shape the model was trained on.
        row = pd.DataFrame({"study_hours": [hours]})
        prediction = model.predict(row)[0]
        probability = model.predict_proba(row)[0][1]
        label = "Pass" if prediction == 1 else "Fail"
        print(
            f"  {hours:>4.1f} study hours -> {label:<4} "
            f"(probability of passing: {probability:.1%})"
        )


def explain_no_retraining():
    """Print the Exercise 1 answer: why the model need not be trained again."""
    print(
        """
Why the model does not need to be trained again
-----------------------------------------------
Training is the expensive part. It reads every student's study hours and
passed/failed outcome and searches for the coefficient and intercept that best
separate the two groups. That search is what produced the numbers above.

joblib.dump() wrote those learned parameters to disk, and joblib.load() reads
them straight back. The reloaded model is mathematically identical to the one
still in memory, so it can predict immediately.

Predicting is just arithmetic on the saved parameters: multiply the study hours
by the coefficient, add the intercept, and pass the result through the logistic
function to get a probability. No data, no search, no fitting involved.

You retrain only when the learned parameters stop being correct — for example
if you add attendance_percentage as a second input, the model needs a second
coefficient, and only training can find it.

(In this dataset the pass threshold sits near 3.0 daily study hours: day that
clearly crosses it and you pass, clearly miss it and you fail.)
"""
    )


def main():
    dataset = generate_dataset()
    train_and_save(dataset)
    load_and_predict()
    explain_no_retraining()


if __name__ == "__main__":
    main()
