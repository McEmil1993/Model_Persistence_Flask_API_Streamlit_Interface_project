"""Exercise 1 - Model Persistence (Rusel)

Trains the Pass/Fail model and saves it to a file with joblib.dump().

Run once:  python train_model.py
"""

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Training data: study hours vs. result (0 = Fail, 1 = Pass)
study_hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

X = pd.DataFrame({"study_hours": study_hours})
y = pd.Series(results, name="result")

MODEL_FILE = "rusel_pass_model.joblib"

model = DecisionTreeClassifier(criterion="gini", random_state=1)
model.fit(X, y)

joblib.dump(model, MODEL_FILE)
print(f"Training finished. Model saved to {MODEL_FILE}")
