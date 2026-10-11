# Exercise 1 - Model Persistence (study hours only)
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# 1. Training data: study hours -> result (0 = Fail, 1 = Pass)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

# 2. Train the model
model = LogisticRegression()
model.fit(X, y)

# 3. SAVE the model with a different filename
MODEL_FILE = "student_pass_model.joblib"
joblib.dump(model, MODEL_FILE)
print(f"Model saved as {MODEL_FILE}")

# 4. LOAD the model (no training happens here)
loaded_model = joblib.load(MODEL_FILE)

# 5. Three predictions using different study hours
for hours in [2, 5, 9]:
    pred = loaded_model.predict([[hours]])[0]
    prob = loaded_model.predict_proba([[hours]])[0][1]
    print(f"Study hours: {hours} -> {'Pass' if pred == 1 else 'Fail'} (pass probability: {prob:.2f})")
