"""Streamlit student outcome demo with a small comparison chart."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_pass_model.joblib"

st.set_page_config(page_title="Student Outcome Predictor", page_icon="📘")
st.title("Student Outcome Predictor")
st.write("Explore how study hours and attendance relate to the model's pass prediction.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` from this folder first.")
    st.stop()

model = joblib.load(MODEL_PATH)
with st.form("student_prediction"):
    student_name = st.text_input("Student name", placeholder="e.g. Jordan")
    study_hours = st.number_input(
        "Study hours per week", min_value=0.0, max_value=100.0, value=15.0, step=1.0
    )
    attendance = st.slider("Attendance percentage", min_value=0, max_value=100, value=80)
    submitted = st.form_submit_button("Predict outcome")

if submitted:
    if not student_name.strip():
        st.warning("Enter a student name to see the result.")
    else:
        student_input = pd.DataFrame(
            [[study_hours, attendance]],
            columns=["study_hours", "attendance_percentage"],
        )
        predicted_class = int(model.predict(student_input)[0])
        label = "Pass" if predicted_class == 1 else "Needs support"
        st.subheader(f"{student_name.strip()}: {label}")
        st.write(
            f"The model used **{study_hours:g} study hours per week** and "
            f"**{attendance}% attendance** to produce this prediction."
        )
        st.caption(
            "How it works: a logistic-regression classifier was fitted on the sample "
            "student records. Its saved scaler and classifier are loaded from the "
            "joblib file; the app supplies the two inputs and displays the predicted class. "
            "This is a learning demo, not a guarantee of an individual outcome."
        )

        # Hold attendance at the entered value and compare classes across study hours.
        hour_values = np.arange(0, 46, 1, dtype=float)
        comparison = pd.DataFrame(
            {"study_hours": hour_values, "attendance_percentage": attendance}
        )
        classes = model.predict(comparison).astype(int)
        figure, axis = plt.subplots(figsize=(8, 3.5))
        axis.step(hour_values, classes, where="mid", color="#2878B5", linewidth=2)
        axis.scatter([study_hours], [predicted_class], color="#D1495B", s=85, zorder=3,
                     label=f"{student_name.strip()} ({label})")
        axis.set_yticks([0, 1], ["Needs support", "Pass"])
        axis.set_xlabel("Study hours per week")
        axis.set_ylabel("Predicted class")
        axis.set_title(f"Predicted class by study hours at {attendance}% attendance")
        axis.set_xlim(0, 45)
        axis.grid(axis="x", alpha=0.2)
        axis.legend(loc="best")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
