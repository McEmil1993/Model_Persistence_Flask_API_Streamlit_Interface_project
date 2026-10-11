import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

data = {
    'study_hours': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0],
    'attendance_percentage': [50, 55, 60, 65, 70, 60, 75, 80, 70, 85, 90, 85, 90, 95, 98],
    'passed': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['study_hours', 'attendance_percentage']]
y = df['passed']

model = LogisticRegression(random_state=42)
model.fit(X, y)
joblib.dump(model, 'student_prediction_project/student_pass_model.joblib')
print("✅ Task 1: Model saved to student_pass_model.joblib!")
