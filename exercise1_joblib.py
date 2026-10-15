"""Exercise 1 - Save and load a model with joblib."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Train and save the model with joblib.dump()
df = pd.read_csv("data.csv")
model = LogisticRegression()
model.fit(df[["study_hours"]], df["passed"])
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")

# 2. Load the model with joblib.load()
loaded_model = joblib.load("model.pkl")
print("Model loaded from model.pkl")

# 3. Three predictions with different study hours (no .fit() called again)
for hours in [1.0, 4.5, 8.0]:
    X = pd.DataFrame({"study_hours": [hours]})
    pred = loaded_model.predict(X)[0]
    prob = loaded_model.predict_proba(X)[0][1]
    label = "Pass" if pred == 1 else "Fail"
    print(f"Study hours = {hours:>4} -> {label} (probability of passing = {prob:.2f})")

# 4. Why no retraining is needed
print("""
Explanation:
Training (fit) finds the model's learned parameters (the coefficient and
intercept). joblib.dump() stores the whole fitted object, including those
parameters, in model.pkl. joblib.load() restores that same object, so it
already 'knows' what it learned. predict() only applies the stored formula
to new input; it does not learn anything. Retraining is needed only when
the data or the features change.
""")
