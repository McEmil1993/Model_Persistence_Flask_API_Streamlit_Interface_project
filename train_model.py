import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# 1. Training data (study hours only)
data = pd.DataFrame({
    "study_hours": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
                    5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
    "passed":      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
})

X = data[["study_hours"]]
y = data["passed"]

# 2. Train
model = LogisticRegression()
model.fit(X, y)

# 3. Save the model (different filename, as required)
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")

# 4. Load the saved model (NO training here)
loaded_model = joblib.load("student_pass_model.joblib")

# 5. Three predictions with different study hours
for hours in [2, 5, 8]:
    new_data = pd.DataFrame({"study_hours": [hours]})
    result = loaded_model.predict(new_data)[0]
    prob = loaded_model.predict_proba(new_data)[0][1]
    print(f"Study hours: {hours} -> {'Pass' if result == 1 else 'Fail'} (pass probability: {prob:.2f})")