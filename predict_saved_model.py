import joblib

model = joblib.load("student_pass_model.joblib")

for h in (1, 3.5, 6):
    pred = model.predict([[h]])[0]
    prob = model.predict_proba([[h]])[0][1]
    print(f"{h} study hours -> {'PASS' if pred == 1 else 'FAIL'} (pass probability {prob:.2f})")

# Why no retraining? joblib.dump() stored the *fitted* model: its learned
# coefficients and intercept. joblib.load() restores that exact object, so
# predict() just applies the stored formula to the new input. Training
# (fit) is only needed when the data or the algorithm changes.
