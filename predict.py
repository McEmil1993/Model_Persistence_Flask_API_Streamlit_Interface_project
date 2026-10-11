import joblib

model = joblib.load("student_pass_model.joblib")
print("Model loaded from disk. No retraining needed.")

samples = [
    (2, 55),
    (5, 75),
    (8, 90),
]

for hours, attendance in samples:
    prediction = int(model.predict([[hours, attendance]])[0])
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours = {hours}, Attendance = {attendance}% -> {result}")