"""Interactive interface for the saved student pass/fail model."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"
FEATURES = ["study_hours", "attendance_percentage"]

# Load the model trained by train_model.py.
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="📚")
st.title("Student Pass/Fail Predictor")
st.write("Enter a student's study time and attendance to estimate the outcome.")

name = st.text_input("Student name", value="Student")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=24.0, value=6.0, step=0.5
)
attendance = st.number_input(
    "Attendance percentage", min_value=0.0, max_value=100.0, value=80.0, step=1.0
)

# Make a prediction only after the button is clicked.
if st.button("Predict result"):
    student = pd.DataFrame(
        [{"study_hours": study_hours, "attendance_percentage": attendance}],
        columns=FEATURES,
    )
    prediction = int(model.predict(student)[0])
    label = "Pass" if prediction == 1 else "Fail"

    st.subheader(f"{name.strip() or 'Student'} is predicted to {label.lower()}.")
    st.write(
        f"This prediction used {study_hours:g} study hours and "
        f"{attendance:g}% attendance. The logistic regression model learned "
        "from the sample student records how study hours and attendance relate "
        "to passing, then used those learned patterns to estimate this result. "
        "The saved model was loaded from student_pass_model.joblib."
    )

    # Show predicted classes across study hours at the entered attendance.
    hours = [value / 2 for value in range(49)]
    chart_data = pd.DataFrame({
        "study_hours": hours,
        "attendance_percentage": [attendance] * len(hours),
    })
    classes = model.predict(chart_data[FEATURES])

    fig, ax = plt.subplots()
    ax.step(hours, classes, where="mid", label="Predicted class")
    ax.scatter([study_hours], [prediction], color="red", s=80, label="This student")
    ax.set_yticks([0, 1], labels=["Fail (0)", "Pass (1)"])
    ax.set_xlabel("Study hours")
    ax.set_ylabel("Predicted class")
    ax.set_title(f"Predictions at {attendance:g}% attendance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
