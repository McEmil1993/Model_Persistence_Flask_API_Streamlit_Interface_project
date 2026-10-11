# train_model.py
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample training data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])  # study hours
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # 0=Fail, 1=Pass

model = LogisticRegression()
model.fit(X, y)

# SAVE with custom name
joblib.dump(model, "student_pass_model.joblib")
print("Model saved!")

# LOAD it back
loaded_model = joblib.load("student_pass_model.joblib")

# 3 predictions
for hours in [2, 5, 8]:
    pred = loaded_model.predict([[hours]])[0]
    print(f"{hours} hours -> {'Pass' if pred == 1 else 'Fail'}")