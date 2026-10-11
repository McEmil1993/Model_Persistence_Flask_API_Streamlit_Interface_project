from pathlib import Path
import joblib
from sklearn.tree import DecisionTreeClassifier

# Example training data: study hours -> predicted class.
# Replace these examples with your instructor's real dataset when available.
X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = ["Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass",
     "Pass", "Pass", "Pass", "Pass"]

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

model_path = Path(__file__).parent / "student_pass_model.joblib"
joblib.dump(model, model_path)
print(f"Model trained and saved to: {model_path}")

# Load the saved model and make three predictions without retraining.
loaded_model = joblib.load(model_path)
for hours in [2, 5, 8]:
    prediction = loaded_model.predict([[hours]])[0]
    print(f"Study hours: {hours} -> Prediction: {prediction}")

print("The saved model already contains the learned decision rules. "
      "Load it with joblib.load() to predict without training again.")
