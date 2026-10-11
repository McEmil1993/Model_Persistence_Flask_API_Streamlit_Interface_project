from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "models" / "student_pass_model.joblib"   # new filename

data = pd.read_csv(BASE_DIR / "data" / "students.csv")
X = data[["study_hours"]]
y = data["passed"]

# 1) TRAIN (slow, done once)
model = LogisticRegression()
model.fit(X, y)

# 2) SAVE
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# 3) LOAD (a brand-new object - no training happens here)
loaded_model = joblib.load(MODEL_FILE)

# 4) PREDICT with three different study hours
for hours in (1.5, 4.5, 8.0):
    sample = pd.DataFrame({"study_hours": [hours]})
    pred = loaded_model.predict(sample)[0]
    prob = loaded_model.predict_proba(sample)[0][1]
    print(f"{hours:>4} hours -> {'PASS' if pred == 1 else 'FAIL'} (pass probability {prob:.0%})")
