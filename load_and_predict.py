import joblib

model = joblib.load("student_pass_model.joblib")

for hours in [2, 4.5, 7]:
    pred = model.predict([[hours]])[0]
    print(f"{hours} study hours -> {'Pass' if pred == 1 else 'Fail'}")