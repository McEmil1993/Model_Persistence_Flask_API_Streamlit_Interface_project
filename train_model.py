"""Train and save both models.

model.pkl     -> uses study hours only (the 'provided' model, used by Exercises 1 and 3)
model_v2.pkl  -> uses study hours + attendance percentage (used by Exercise 2)
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

# Model 1: study hours only
m1 = LogisticRegression()
m1.fit(df[["study_hours"]], df["passed"])
joblib.dump(m1, "model.pkl")

# Model 2: study hours + attendance
features = ["study_hours", "attendance_percentage"]
m2 = LogisticRegression(max_iter=1000)
m2.fit(df[features], df["passed"])
joblib.dump(m2, "model_v2.pkl")

print("Saved model.pkl (1 feature) and model_v2.pkl (2 features)")
print("Accuracy model.pkl   :", round(m1.score(df[["study_hours"]], df["passed"]), 3))
print("Accuracy model_v2.pkl:", round(m2.score(df[features], df["passed"]), 3))
