import joblib
from sklearn.tree import DecisionTreeClassifier

# Example training data: [study_hours]
# Labels: 0 = Needs Improvement, 1 = Likely to Pass
X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved as student_pass_model.joblib")

# Load the saved model and predict without training again
loaded_model = joblib.load("student_pass_model.joblib")

for hours in [2, 5, 8]:
    prediction = loaded_model.predict([[hours]])[0]
    label = "Likely to Pass" if prediction == 1 else "Needs Improvement"
    print(f"Study hours: {hours} -> Prediction: {label}")
