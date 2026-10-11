
import joblib
from sklearn.tree import DecisionTreeClassifier

# Training data
# 0 = Fail, 1 = Pass
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")

# Make three predictions
for hours in [2, 5, 7]:
    prediction = loaded_model.predict([[hours]])[0]
    result = "Pass" if prediction == 1 else "Fail"
    print(f"{hours} study hours: {result}")
