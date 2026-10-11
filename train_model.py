
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = {
    "StudyHours": [
        1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
        6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
        1, 3, 4, 6, 7, 8, 9, 11, 12, 12
    ],
    "Result": [
        "Fail", "Fail", "Fail", "Fail", "Fail",
        "Fail", "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass",
        "Fail", "Fail", "Fail", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

X = df[["StudyHours"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=3))
])

model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)

joblib.dump(model, "student_pass_model.joblib")

print("Model training completed.")
print(f"Test accuracy: {accuracy:.2%}")
print("Model saved successfully.")

loaded_model = joblib.load("student_pass_model.joblib")

new_students = pd.DataFrame({
    "StudyHours": [2, 6, 10]
})

predictions = loaded_model.predict(new_students)

print("\nPredictions for new students:")

for hours, result in zip(
    new_students["StudyHours"],
    predictions
):
    print(f"Study Hours: {hours} | Prediction: {result}")
