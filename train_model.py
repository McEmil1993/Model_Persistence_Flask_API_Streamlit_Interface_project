import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# STEP 1: Create sample training data
data = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})

# STEP 2: Separate input and target
X = data[["study_hours"]]
y = data["passed"]

# STEP 3: Train the model
model = LogisticRegression()
model.fit(X, y)

print("Model training completed!")

# STEP 4: Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model saved successfully!")

# STEP 5: Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")

print("Model loaded successfully!")

# STEP 6: Make three predictions
test_hours = [2, 6, 9]

for hours in test_hours:
    input_data = pd.DataFrame({
        "study_hours": [hours]
    })

    prediction = loaded_model.predict(input_data)[0]

    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(f"Study hours: {hours} -> Prediction: {result}")