import joblib

model = joblib.load("student_pass_model.joblib")

for hours in [2, 5, 8]:
    prediction = model.predict([[hours]])
    print(f"Study Hours: {hours}")
    print(f"Prediction: {prediction[0]}")
    print()