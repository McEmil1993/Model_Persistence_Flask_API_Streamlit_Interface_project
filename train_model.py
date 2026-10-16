import joblib
from sklearn.linear_model import LogisticRegression

# 1. Prepare the training data
# X = study hours
# y = 0 means Fail, 1 means Pass

X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

# 2. Create and train the model
model = LogisticRegression()
model.fit(X, y)

print("Model training completed!")

# 3. Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model saved successfully!")

# 4. Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")

print("Model loaded successfully!")

# 5. Make three predictions
study_hours_list = [2, 5, 8]

for hours in study_hours_list:
    prediction = loaded_model.predict([[hours]])[0]

    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(f"Study hours: {hours} -> Prediction: {result}")