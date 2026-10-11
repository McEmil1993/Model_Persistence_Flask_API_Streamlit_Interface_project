import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_FILENAME = "student_pass_model.joblib"

EXPLANATION = (
    "Why the model does not need to be trained again:\n"
    "  - Training learns the relationship between study hours and pass/fail and\n"
    "    stores the result (the coefficients and intercept) inside the model object.\n"
    "  - joblib.dump() serializes that already-learned object to disk, so the\n"
    "    trained weights survive after the script stops running.\n"
    "  - joblib.load() deserializes the file back into an identical model object.\n"
    "  - Prediction (model.predict) only applies the stored weights to new input,\n"
    "    it does NOT re-learn from data, so retraining is unnecessary and would\n"
    "    simply reproduce the same weights.\n"
    "  - Retraining is only required when the training data changes or when new\n"
    "    input features (such as attendance) are added to the model."
)


def build_training_data():
    study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float).reshape(-1, 1)
    passed = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int)
    return study_hours, passed


def train_and_save(filename=MODEL_FILENAME):
    study_hours, passed = build_training_data()
    model = LogisticRegression()
    model.fit(study_hours, passed)
    joblib.dump(model, filename)
    print(f"Model trained and saved to '{filename}'")
    return model


def load_model(filename=MODEL_FILENAME):
    model = joblib.load(filename)
    print(f"Model loaded from '{filename}'")
    return model


def make_predictions(model, hours_list):
    print("\nPredictions using the loaded model:")
    for hours in hours_list:
        prediction = int(model.predict([[hours]])[0])
        pass_probability = float(model.predict_proba([[hours]])[0][1])
        label = "PASS" if prediction == 1 else "FAIL"
        print(f"  Study hours: {hours:>4} -> {label:<4} (pass probability: {pass_probability:.1%})")


def main():
    train_and_save()
    model = load_model()
    make_predictions(model, [2, 5, 8])
    print("\n" + EXPLANATION)


if __name__ == "__main__":
    main()
