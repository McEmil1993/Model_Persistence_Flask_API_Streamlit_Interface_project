import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    "study_hours":           [1,2,3,4,5,6,7,8,9,10,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,0.5],
    "attendance_percentage": [55,60,50,70,65,75,80,85,90,95,40,65,60,72,78,70,88,92,96,45],
    "passed":                [0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0],
})

X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "student_pass_model_v2.joblib")
print("Model saved as student_pass_model_v2.joblib")
print("Training accuracy:", model.score(X, y))