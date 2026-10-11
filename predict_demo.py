import joblib
import pandas as pd

# Load the saved model (no training needed)
model = joblib.load("student_pass_model.joblib")

# Three predictions with different study hours
test_hours = [2, 5, 9]

for hours in test_hours:
    X_new = pd.DataFrame([[hours]], columns=["study_hours"])
    prediction = model.predict(X_new)[0]
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours} -> Prediction: {result}")