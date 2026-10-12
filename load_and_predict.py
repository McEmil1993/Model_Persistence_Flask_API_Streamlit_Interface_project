import joblib
import pandas as pd

model = joblib.load("student_pass_model.joblib")

for hours in [2, 5, 9]:
    pred = model.predict(pd.DataFrame({"study_hours": [hours]}))[0]
    print(f"{hours} study hours -> {'Pass' if pred == 1 else 'Fail'}")