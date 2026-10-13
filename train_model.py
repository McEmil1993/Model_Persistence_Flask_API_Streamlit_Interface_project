"""Train and persist the student pass/fail classifier.

Dataset: synthetic table with two columns -- weekly study hours and
attendance percentage -- mapped to a binary pass/fail label.
The fitted sklearn pipeline is stored with joblib so the Flask and
Streamlit apps can reuse it without refitting.
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_FILE = Path(__file__).with_name("student_pass_model.joblib")

log = logging.getLogger(__name__)


def make_student_data(n_rows: int = 400, seed: int = 11):
    """Create synthetic (hours, attendance) features and pass/fail targets."""
    gen = np.random.default_rng(seed)
    hours = gen.uniform(0.0, 10.0, n_rows)
    attendance_pct = gen.uniform(0.0, 100.0, n_rows)

    hours_weight = 0.8
    attendance_weight = 0.05
    noise = gen.normal(0.0, 0.5, n_rows)
    combined = hours_weight * hours + attendance_weight * attendance_pct + noise
    passed = (combined > 5.0).astype(int)

    features = np.column_stack([hours, attendance_pct])
    return features, passed


def train_classifier(features, targets):
    """Fit a scaled logistic-regression pipeline and report split scores."""
    x_train, x_eval, y_train, y_eval = train_test_split(
        features, targets, test_size=0.25, random_state=7
    )
    # Hours (0-10) and attendance (0-100) live on different scales,
    # so standardising first keeps one feature from dominating.
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(x_train, y_train)
    log.info("train score: %.3f", clf.score(x_train, y_train))
    log.info("held-out score: %.3f", clf.score(x_eval, y_eval))
    return clf


def demo_predictions(clf):
    """Run three sample students with distinct study-hour values."""
    samples = [
        (1.5, 55.0),
        (4.5, 80.0),
        (9.0, 95.0),
    ]
    for hrs, attend in samples:
        chance = float(clf.predict_proba([[hrs, attend]])[0][1])
        outcome = "PASS" if chance >= 0.5 else "FAIL"
        log.info(
            "hours=%.1f attendance=%.0f%% -> %s (P(pass)=%.2f)",
            hrs,
            attend,
            outcome,
            chance,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pass/fail model.")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model-out", type=Path, default=MODEL_FILE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    features, targets = make_student_data(n_rows=args.samples, seed=args.seed)
    clf = train_classifier(features, targets)

    # Task 1: persist the fitted pipeline.
    joblib.dump(clf, args.model_out)
    log.info("saved model -> %s", args.model_out.name)

    # Task 2: reload it to prove persistence works.
    reloaded = joblib.load(args.model_out)
    log.info("reloaded model <- %s", args.model_out.name)

    # Task 3: three predictions with different study hours.
    demo_predictions(reloaded)

    # Task 4: why refitting is unnecessary.
    log.info(
        "No refit needed: the dump file already holds the fitted scaler "
        "(mean/scale) plus the trained regression weights, so loading "
        "restores a ready-to-use predictor."
    )


if __name__ == "__main__":
    main()
