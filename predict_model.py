
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Three different study hours
study_hours_list = [2, 5, 8]

for hours in study_hours_list:
    input_data = pd.DataFrame({
        "study_hours": [hours]
    })

    prediction = model.predict(input_data)[0]

    result = "Pass" if prediction == 1 else "Fail"

    print(f"Study Hours: {hours}")
    print(f"Prediction: {result}")
    print("-" * 25)
