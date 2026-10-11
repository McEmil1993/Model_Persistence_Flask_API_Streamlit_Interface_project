
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Create the dataset
data = {
    "study_hours": [
        1.0, 1.5, 2.0, 2.5, 3.0,
        3.5, 4.0, 5.0, 5.5, 6.0
    ],
    "passed": [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# 2. Select the input and target
X = df[["study_hours"]]
y = df["passed"]

# 3. Create the model
model = LogisticRegression()

# 4. Train the model
model.fit(X, y)

# 5. Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained and saved successfully!")

# 6. Make three predictions
test_data = pd.DataFrame({
    "study_hours": [2.0, 4.0, 6.0]
})

predictions = model.predict(test_data)

for hours, prediction in zip(
    test_data["study_hours"], predictions
):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours} | Prediction: {result}")
