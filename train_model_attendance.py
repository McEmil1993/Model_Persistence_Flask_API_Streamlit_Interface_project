"""Exercise 2 - Retrain with BOTH study hours and attendance percentage."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_FILENAME = "student_pass_model_attendance.joblib"


def make_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 10, n)
    attendance = rng.uniform(40, 100, n)
    score = 0.7 * (hours - 5) + 0.06 * (attendance - 70)
    prob = 1 / (1 + np.exp(-score))
    passed = (rng.random(n) < prob).astype(int)
    return np.column_stack([hours, attendance]), passed


def main():
    X, y = make_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")


if __name__ == "__main__":
    main()
