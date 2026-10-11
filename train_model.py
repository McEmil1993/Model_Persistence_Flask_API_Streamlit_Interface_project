from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

def train_and_save_model():
    # Small sample dataset for classroom demonstration only.
    # Each number is study hours; labels are 0 = Fail and 1 = Pass.
    X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH.name}")
    return model

def load_model():
    if not MODEL_PATH.exists():
        print("Saved model not found. Training it now...")
        return train_and_save_model()
    print(f"Loaded saved model: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    model = load_model()
    print("\nThree example predictions:")
    for hours in [2, 5, 8]:
        prediction = model.predict([[hours]])[0]
        label = "Pass" if prediction == 1 else "Fail"
        print(f"Study hours: {hours} -> Prediction: {label}")

    print("\nThe saved model can be loaded again without retraining.")
    print("It only needs to be retrained when you change the training data or model.")
