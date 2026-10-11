import joblib
from sklearn.tree import DecisionTreeClassifier

# Sample student training data

# 0 = At risk of failing

# 1 = Likely to pass

X = [[0], [1], [2], [3], [4],
[5], [6], [7], [8], [9]]

y = [0, 0, 0, 0, 0,
1, 1, 1, 1, 1]

# Train the model

print("Training the model...")

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the trained model

joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved successfully!")

# Load the saved model

loaded_model = joblib.load("student_pass_model.joblib")
print("Saved model loaded successfully!")

# Make three predictions

test_hours = [[2], [5], [8]]
predictions = loaded_model.predict(test_hours)

for student, prediction in zip(test_hours, predictions):
 if prediction == 1:
  result = "Likely to pass"
 else:
  result = "At risk of failing"


print(f"Study hours: {student[0]} -> {result}")


print("Exercise 1 completed!")
