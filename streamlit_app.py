
"""Simple Streamlit interface for the student pass prediction model."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Predictor", page_icon="📚")
st.title("Student Pass Predictor")
st.write(
    "Estimate a student's result using study hours and attendance percentage."
)

if not MODEL_PATH.exists():
    st.error(
        "The model file is missing. "
        "Run `python train_model.py` and refresh this page."
    )
    st.stop()


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


model = get_model()

with st.form("prediction_form"):
    name = st.text_input("Student name")

    study_hours = st.number_input(
        "Study hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    attendance_percentage = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    submitted = st.form_submit_button("Predict")


if submitted:
    # Predict using both features
    prediction = int(
        model.predict([
            [study_hours, attendance_percentage]
        ])[0]
    )

    student = name.strip() or "Student"

    if prediction == 1:
        st.success(f"{student}, the model predicts that you will pass.")
    else:
        st.info(
            f"{student}, the model predicts that "
            "you may need more study time."
        )

    st.caption(
        "This prediction uses the previously trained model. "
        "The model is not retrained for each prediction. "
        "Results are estimates based on example training data."
    )

  
    sample_hours = np.arange(0, 13, 1, dtype=float)


    sample_data = np.column_stack((
        sample_hours,
        np.full(len(sample_hours), attendance_percentage)
    ))

    classes = model.predict(sample_data)

    fig, ax = plt.subplots()

    ax.step(
        sample_hours,
        classes,
        where="mid",
        color="#2563eb",
        linewidth=2,
        label="Predicted result"
    )

    ax.scatter(
        [study_hours],
        [prediction],
        color="#f97316",
        s=80,
        zorder=3,
        label="Your prediction"
    )

    ax.set(
        xlabel="Study hours",
        ylabel="Predicted class (0 = needs more study, 1 = pass)",
        title=f"Prediction at {attendance_percentage:.0f}% attendance",
        yticks=[0, 1],
        xlim=(0, 12),
        ylim=(-0.15, 1.15)
    )

    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)
