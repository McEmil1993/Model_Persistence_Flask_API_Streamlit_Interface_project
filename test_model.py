import joblib
import pandas as pd

# 1. Load the previously saved model.
model = joblib.load("student_pass_model.joblib")

print("Saved model loaded successfully!\n")

# 2. Test three different study-hour values.
study_hours_list = [2, 5, 8]

# 3. Make a prediction for each value.
for hours in study_hours_list:

    # Use the same feature name as during training.
    student_data = pd.DataFrame({
        "study_hours": [hours]
    })

    prediction = model.predict(student_data)[0]

    print(f"Study Hours: {hours}")
    print(f"Predicted Result: {prediction}")
    print("-" * 30)

print("\nAll three predictions are complete.")
