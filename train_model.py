
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# 1. SAMPLE TRAINING DATA
# Features: [study_hours, attendance_percentage]
# Labels: 0 = Fail, 1 = Pass
# --------------------------------------------------

X = [
    [1, 40], [2, 50], [2, 60], [3, 55],
    [1, 70], [3, 65], [4, 60], [2, 80],
    [4, 75], [5, 70], [3, 85], [4, 90],
    [5, 80], [6, 75], [5, 90], [6, 85],
    [7, 80], [6, 95], [8, 90], [7, 95],
    [2, 95], [3, 95], [1, 95], [4, 98],
    [5, 45], [6, 50], [7, 55], [8, 60],
    [1, 30], [2, 35], [3, 40], [4, 45],
    [5, 55], [6, 60], [7, 65], [8, 70],
    [3, 75], [4, 80], [5, 85], [6, 90]
]

y = [
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 0, 1,
    0, 0, 0, 1,
    0, 0, 0, 0,
    0, 0, 1, 1,
    0, 1, 1, 1
]

# --------------------------------------------------
# 2. SPLIT DATA INTO TRAINING AND TESTING SETS
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. CREATE AND TRAIN THE MODEL
# --------------------------------------------------

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 4. EVALUATE THE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print(f"Model accuracy on test sample: {accuracy:.2%}")

# --------------------------------------------------
# 5. SAVE THE TRAINED MODEL
# --------------------------------------------------

model_path = Path(__file__).parent / "student_pass_model.joblib"

joblib.dump(model, model_path)

print(f"Model saved successfully to: {model_path}")
print("Model features: Study Hours and Attendance Percentage")
