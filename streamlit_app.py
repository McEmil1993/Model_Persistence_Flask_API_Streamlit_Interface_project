from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Predictor")
st.title("Student Pass Predictor")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run train_model.py before starting this app.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Student inputs
student_name = st.text_input("Student name")

study_hours = st.number_input(
    "Study hours per day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
)

attendance_percentage = st.number_input(
    "Attendance percentage",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=1.0,
)

# Prediction
if st.button("Predict result", type="primary"):
    predicted_class = int(model.predict([[study_hours]])[0])
    label = "Pass" if predicted_class == 1 else "Fail"
    name = student_name.strip() or "The student"

    st.success(f"{name}: predicted result is {label}.")

    st.write(f"**Study hours:** {study_hours:g} hours per day")
    st.write(f"**Attendance:** {attendance_percentage:g}%")
    st.write(f"**Predicted class:** {predicted_class}")

    st.info(
        "The prediction was generated using the saved machine learning "
        "model and study hours only. Attendance is displayed for reference "
        "but does not affect the prediction."
    )

# Chart
hours_range = np.linspace(0, 24, 49)

chart_data = pd.DataFrame({
    "Study hours per day": hours_range,
    "Predicted class (0 = Fail, 1 = Pass)": model.predict(
        hours_range.reshape(-1, 1)
    ),
})

st.subheader("Predicted class by study time")
st.line_chart(chart_data, x="Study hours per day")