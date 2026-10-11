import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_PATH = BASE_DIR / "models" / "student_pass_model.joblib"

# Load the dataset
data = pd.read_csv(DATA_PATH)

X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

# Split data for training and evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)
model.fit(X_train, y_train)

print("Model trained successfully!")
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save the trained model
joblib.dump(model, MODEL_PATH)
print("Model saved:", MODEL_PATH)

# Load the saved model
loaded_model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Make three predictions using the loaded model
for hours, attendance in [(2, 60), (5, 80), (8, 95)]:
    prediction = loaded_model.predict(
        pd.DataFrame(
            [[hours, attendance]],
            columns=["study_hours", "attendance_percentage"]
        )
    )[0]

    label = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours}, Attendance: {attendance}%, Prediction: {label}")

# The saved model retains its learned parameters, so it can make
# predictions without being trained again.