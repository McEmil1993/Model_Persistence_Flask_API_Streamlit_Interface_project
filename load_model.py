
import joblib

MODEL_PATH = "caitom_pass_model.joblib"


def load_saved_model():
    return joblib.load(MODEL_PATH)


def predict_student(model, hours):
    prediction = model.predict([[hours]])[0]
    return "PASS" if prediction == 1 else "FAIL"


if __name__ == "__main__":
    model = load_saved_model()

    print("Saved model loaded successfully.")
    print("\nSTUDENT PREDICTIONS")
    print("-------------------")

    for hours in [2, 5, 7]:
        result = predict_student(model, hours)
        print(f"Study Hours: {hours} | Prediction: {result}")

    print("\nNo retraining was needed.")
