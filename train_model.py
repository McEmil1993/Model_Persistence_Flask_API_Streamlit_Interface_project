from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression

# Example training data for practice. The model uses study hours only.
# Replace these example values with your instructor's dataset if one was provided.
study_hours = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
results = ["Fail", "Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass"]

model = LogisticRegression()
model.fit(study_hours, results)

output_path = Path(__file__).with_name("student_pass_model.joblib")
joblib.dump(model, output_path)
print(f"Model trained and saved to: {output_path.name}")

# Demonstrate loading the saved model (no retraining required for predictions).
loaded_model = joblib.load(output_path)
for hours in [2, 5, 8]:
    prediction = loaded_model.predict([[hours]])[0]
    print(f"Study hours: {hours} -> Prediction: {prediction}")