"""Exercise 1 - Train a study-hours model and save it as student_pass_model.joblib."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_FILENAME = "student_pass_model.joblib"  # changed filename (Exercise 1)


def make_data(n=200, seed=42):
    """Synthetic data: more study hours -> higher chance of passing."""
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 10, n)
    prob = 1 / (1 + np.exp(-(hours - 5)))
    passed = (rng.random(n) < prob).astype(int)
    return hours.reshape(-1, 1), passed


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
