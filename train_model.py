import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Training data (study hours -> 0 = Fail, 1 = Pass)
data = pd.DataFrame({
    "study_hours": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8],
    "passed":      [0,   0, 0,   0, 0,   0, 1,   1, 1,   1, 1,   1, 1, 1],
})

X = data[["study_hours"]]
y = data["passed"]

# 2. Train the model
model = LogisticRegression(C=10)
model.fit(X, y)

# 3. SAVE the model using a different filename
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")

# 4. LOAD the model back
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully")

# 5. Make three predictions with different study hours
test_hours = [1.0, 5.0, 8.0]
for h in test_hours:
    sample = pd.DataFrame({"study_hours": [h]})
    prediction = loaded_model.predict(sample)[0]
    probability = loaded_model.predict_proba(sample)[0][1]
    label = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {h} -> {label} (pass probability: {probability:.2f})")