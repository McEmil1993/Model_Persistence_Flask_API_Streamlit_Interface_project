"""Exercise 1 — Model Persistence.

Trains a simple student pass/fail classifier on study hours,
saves it as `student_pass_model.joblib`, reloads it, and makes
three predictions without retraining.
"""
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_FILENAME = "student_pass_model.joblib"


def main():
    # 1. Training data: [study_hours] -> pass (0 = Fail, 1 = Pass)
    X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    # 2. Train
    model = LogisticRegression()
    model.fit(X, y)
    print("Model trained.")

    # 3. Save with joblib.dump()
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")

    # 4. Load with joblib.load()
    loaded_model = joblib.load(MODEL_FILENAME)
    print(f"Model loaded from {MODEL_FILENAME}")

    # 5. Three predictions using different study hours
    test_hours = [2, 5, 8]
    for hours in test_hours:
        pred = loaded_model.predict([[hours]])[0]
        proba = loaded_model.predict_proba([[hours]])[0].max()
        label = "Pass" if pred == 1 else "Fail"
        print(f"Study hours={hours} -> prediction={pred} ({label}), confidence={proba:.2f}")

    # 6. Explanation
    print(
        "\nWhy no retraining is needed:\n"
        "joblib.dump() serializes the fitted model object — including its learned "
        "coefficients, intercept, and classes — to disk. joblib.load() restores that "
        "exact fitted object into memory, so predict()/predict_proba() work immediately. "
        "Training (fit) only has to happen once; inference reuses the stored parameters."
    )


if __name__ == "__main__":
    main()
