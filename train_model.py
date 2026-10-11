# train_model.py
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
    "passed":      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
df.to_csv("student_data.csv", index=False)
print("✅ Dataset created: student_data.csv")

X = df[["study_hours"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"✅ Model accuracy: {accuracy_score(y_test, y_pred):.2f}")

joblib.dump(model, "student_pass_model.joblib")
print("✅ Model saved as student_pass_model.joblib")