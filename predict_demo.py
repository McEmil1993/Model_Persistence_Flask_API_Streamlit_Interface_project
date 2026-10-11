import joblib
import pandas as pd

model = joblib.load("student_pass_model.joblib")

for hours in [2, 5, 9]:
    X = pd.DataFrame({"study_hours": [hours]})
    pred = model.predict(X)[0]
    print(f"{hours} study hours -> {'Pass' if pred == 1 else 'Fail'}")