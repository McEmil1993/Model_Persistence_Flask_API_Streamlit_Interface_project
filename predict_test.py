import joblib

# Exercise 1: load the saved model (no retraining needed)
model = joblib.load("student_pass_model.joblib")

# Three predictions with different study hours (and attendance)
tests = [(2, 50), (5, 75), (8, 95)]
for hours, att in tests:
    result = model.predict([[hours, att]])[0]
    print(f"{hours} hours, {att}% attendance -> {'PASS' if result == 1 else 'FAIL'}")

# Why no retraining? The learned values are stored inside the .joblib file.
# joblib.load() restores the model exactly as it was after training.
