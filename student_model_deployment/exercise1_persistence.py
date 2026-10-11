from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run 'python train_model.py' first."
    )

# Load the saved model; no model.fit() or retraining is done here.
model = joblib.load(MODEL_PATH)

study_hours_list = [2, 4, 7]

print("EXERCISE 1 — MODEL PERSISTENCE")
for hours in study_hours_list:
    prediction = model.predict([[hours]])[0]
    print(f"Study hours: {hours} -> Prediction: {prediction}")

print(
    "\nThe model does not need to be trained again because the trained "
    "model was saved to a file and loaded with joblib.load()."
)
