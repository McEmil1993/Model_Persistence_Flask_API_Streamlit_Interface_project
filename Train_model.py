
import joblib
from sklearn.tree import DecisionTreeClassifier

# 1. Sample training data
# X = study hours; y = 0 means Fail, 1 means Pass
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]


model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)


joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")


loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")


study_hours_list = [[2], [5], [7]]
predictions = loaded_model.predict(study_hours_list)

for hours, prediction in zip(study_hours_list, predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours[0]} -> {result}")
