import joblib
import pandas as pd

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Test three different study hours
study_hours = [2.0, 4.0, 6.0]

# Prepare the input data
X_test = pd.DataFrame({
    "study_hours": study_hours
})

# Make predictions
predictions = model.predict(X_test)

# Display results
for hours, prediction in zip(study_hours, predictions):
    print(f"Study hours: {hours}")
    print(f"Prediction: {prediction}")
    print("--------------------")