import joblib
import pandas as pd

model = joblib.load("student_pass_model.joblib")

for hours in [1, 5, 8]:
    X = pd.DataFrame({"study_hours": [hours]})
    result = "Pass" if model.predict(X)[0] == 1 else "Fail"
    print(hours, "hours ->", result)