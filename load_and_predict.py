import joblib

model = joblib.load("student_pass_model.joblib")

for hours in [1, 4.5, 8]:
    pred = model.predict([[hours]])[0]
    prob = model.predict_proba([[hours]])[0][1]
    print(f"{hours} hours -> {'Pass' if pred == 1 else 'Fail'} (pass probability: {prob:.2f})")