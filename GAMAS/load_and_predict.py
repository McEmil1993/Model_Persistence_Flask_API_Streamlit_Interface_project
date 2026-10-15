
import joblib
import pandas as pd

model = joblib.load("student_pass_model.joblib")                # <-- Task 2

for hours in [1, 4, 7]:                                         # <-- Task 3
    X = pd.DataFrame({"study_hours": [hours]})
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    print(f"{hours} study hours -> {'PASS' if pred == 1 else 'FAIL'} (pass probability {prob:.2f})")