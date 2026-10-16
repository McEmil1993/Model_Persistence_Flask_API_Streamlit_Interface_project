from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'dataset.csv'
MODEL_PATH = BASE_DIR / 'student_pass_model.joblib'

# The model deliberately uses study_hours as its sole feature. Attendance is
# collected by the API but is not part of this sample model's training data.
data = pd.read_csv(DATASET_PATH, comment="#")
X = data[['study_hours']]
y = data['passed']

model = LogisticRegression(random_state=42)
model.fit(X, y)
joblib.dump(model, MODEL_PATH)

# Demonstrate persistence: reload the fitted estimator and predict without
# fitting it again.
loaded_model = joblib.load(MODEL_PATH)
for hours in [2, 5, 8]:
    predicted_class = loaded_model.predict(pd.DataFrame({'study_hours': [hours]}))[0]
    label = 'Pass' if str(predicted_class) == '1' else 'Fail'
    print(f'Study hours: {hours}, Prediction: {label}')

print(f'Saved model to {MODEL_PATH.name}')

