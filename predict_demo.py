import joblib

model = joblib.load("student_pass_model.joblib")

for h in (1.5, 5.0, 8.0):
    pred = model.predict([[h]])[0]
    prob = model.predict_proba([[h]])[0][1]
    print(f"{h:>4} study hours -> {'PASS' if pred == 1 else 'FAIL'} (P(pass) = {prob:.2f})")
