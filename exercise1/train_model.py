"""Exercise 1 - Model Persistence (Rayla)

Trains the Pass/Fail model on a NumPy array and saves it with joblib.dump().

Run once:  python train_model.py
"""

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# X = study hours, y = result (0 = Fail, 1 = Pass)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

MODEL_FILE = "rayla_pass_model.joblib"

model = DecisionTreeClassifier(criterion="entropy", random_state=7)
model.fit(X, y)

joblib.dump(model, MODEL_FILE)
print(f"Training finished. Model saved to {MODEL_FILE}")
