
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Create training data
data = {
    "study_hours": [
        1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4, 4.5,
        5, 5, 5.5, 6, 6, 6.5, 7, 7.5, 8, 9, 10
    ],
    "passed": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# 2. Separate features and target
X = df[["study_hours"]]
y = df["passed"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model training completed!")

# 5. Check model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")

# 6. Save the trained model
filename = "student_pass_model.joblib"
joblib.dump(model, filename)

print(f"Model saved as {filename}")

# 7. Load the saved model
loaded_model = joblib.load(filename)

print("Model loaded successfully!")

# 8. Make three predictions using different study hours
study_hours = [2, 5, 8]

for hours in study_hours:
    input_data = pd.DataFrame({"study_hours": [hours]})

    prediction = loaded_model.predict(input_data)[0]
    probability = loaded_model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(
        f"Study hours: {hours} | "
        f"Prediction: {result} | "
        f"Pass probability: {probability:.2%}"
    )

print("\nExercise 1 completed!")
