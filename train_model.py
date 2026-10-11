import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = {
    "study_hours": [
        1, 2, 3, 4, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 5, 6, 7, 8, 9, 10
    ],
    "result": [
        "Fail", "Fail", "Fail", "Fail",
        "Fail", "Fail", "Fail", "Fail",
        "Pass", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

X = df[["study_hours"]]
y = df["result"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully.")

loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully.")

test_data = pd.DataFrame({
    "study_hours": [2, 5, 8]
})

predictions = loaded_model.predict(test_data)

print("\nSTUDENT RESULT PREDICTIONS")
for hours, result in zip(test_data["study_hours"], predictions):
    print(f"Study Hours: {hours} - Prediction: {result}")

print("\nThe model was loaded from the saved file.")
print("It does not need to be trained again for these predictions.")