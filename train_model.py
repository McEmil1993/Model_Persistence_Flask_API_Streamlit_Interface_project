import joblib
from sklearn.tree import DecisionTreeClassifier


X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved successfully!")


loaded_model = joblib.load("student_pass_model.joblib")


test_hours = [[2], [4], [7]]

print("\nThree predictions:")

for hours in test_hours:
    prediction = loaded_model.predict([hours])[0]
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours[0]} -> Prediction: {result}")