"""Streamlit interface for student pass prediction."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")
st.write("Estimate a student's pass/fail class using study hours.")

if not MODEL_PATH.exists():
    st.error("Model file not found. In this folder, run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    student_name = st.text_input("Student name", placeholder="Enter student name")
    study_hours = st.number_input(
        "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
    )
    submitted = st.form_submit_button("Predict result")

if submitted:
    prediction = int(model.predict([[study_hours]])[0])
    pass_probability = float(model.predict_proba([[study_hours]])[0][1])
    label = "Pass" if prediction == 1 else "Fail"
    display_name = student_name.strip() or "Student"

    if prediction == 1:
        st.success(f"{display_name}: predicted result — {label}")
    else:
        st.warning(f"{display_name}: predicted result — {label}")

    st.metric("Attendance Percentage", f"{pass_probability:.1%}")
    st.caption(
        f"How it was generated: the saved machine-learning model received "
        f"{study_hours:g} study hours and returned class {prediction} ({label}). "
        "This is an educational estimate, not a guarantee of actual academic performance."
    )

    # Bonus chart: show the entered study hours and the predicted class.
    fig, ax = plt.subplots()
    ax.bar(["Study hours", "Predicted class"], [study_hours, prediction])
    ax.set_ylabel("Value (hours / class)")
    ax.set_title("Study hours compared with predicted class (0 = Fail, 1 = Pass)")
    ax.set_ylim(0, max(2, study_hours + 1))
    st.pyplot(fig)
    plt.close(fig)

st.info(
    "This sample model uses study hours only. The Flask API accepts attendance "
    "percentage, but attendance will not influence predictions unless the model "
    "is retrained using both features."
)
