"""Streamlit interface for the saved student pass/fail model."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="🎓")
st.title("Student Pass/Fail Predictor")
st.write("Enter a student's details to estimate their likelihood of passing.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "The saved model was not found. Run `python train_model.py` in this "
        "folder first."
    )
    st.stop()

model = load_model()

name = st.text_input("Student name", placeholder="Enter a name")
hours = st.number_input(
    "Study hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
)
attendance = st.slider("Attendance percentage", 0, 100, 75)

if st.button("Predict"):
    features = pd.DataFrame(
        [
            {
                "study_hours": hours,
                "attendance_percentage": attendance,
            }
        ]
    )
    prediction = int(model.predict(features)[0])
    pass_class_index = list(model.classes_).index(1)
    pass_probability = float(model.predict_proba(features)[0][pass_class_index])
    result = "Pass" if prediction == 1 else "Fail"
    display_name = name.strip() or "Student"

    if prediction == 1:
        st.success(
            f"Prediction for {display_name}: {result} "
            f"({pass_probability:.1%} estimated pass probability)"
        )
    else:
        st.warning(
            f"Prediction for {display_name}: {result} "
            f"({pass_probability:.1%} estimated pass probability)"
        )

    st.markdown(
        "**How this prediction was generated:** the app loaded the saved "
        "Logistic Regression model and supplied the study-hours and "
        "attendance values. The model returns a class prediction and an "
        "estimated pass probability; it is not retrained for each prediction."
    )

    # Show how the prediction changes with study hours while holding
    # attendance at the value entered above.
    hours_range = np.linspace(0, 24, 121)
    chart_features = pd.DataFrame(
        {
            "study_hours": hours_range,
            "attendance_percentage": [attendance] * len(hours_range),
        }
    )
    pass_probabilities = model.predict_proba(chart_features)[:, pass_class_index]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours_range, pass_probabilities, color="#2563eb", linewidth=2)
    ax.fill_between(
        hours_range,
        pass_probabilities,
        0.5,
        where=pass_probabilities >= 0.5,
        interpolate=True,
        color="#22c55e",
        alpha=0.16,
    )
    ax.fill_between(
        hours_range,
        pass_probabilities,
        0.5,
        where=pass_probabilities < 0.5,
        interpolate=True,
        color="#f97316",
        alpha=0.14,
    )
    ax.axhline(0.5, color="#64748b", linestyle="--", linewidth=1, label="0.5 threshold")
    ax.scatter(
        [hours],
        [pass_probability],
        color="#111827",
        zorder=3,
        label=f"This student ({hours:g} hours)",
    )
    ax.set(
        xlim=(0, 24),
        ylim=(0, 1),
        xlabel="Study hours",
        ylabel="Estimated probability of passing",
        title=f"Prediction by study hours (attendance held at {attendance}%)",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
