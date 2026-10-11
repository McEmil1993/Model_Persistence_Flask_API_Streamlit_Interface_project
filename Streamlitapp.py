
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "student_pass_model.joblib"
MAX_HOURS = 12

FAIL_LABEL = "Fail (0)"
PASS_LABEL = "Pass (1)"


st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, study_hours: float, attendance_percentage: float):
    X_new = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance_percentage": [attendance_percentage],
    })

    predicted_class = int(model.predict(X_new)[0])
    pass_probability = float(model.predict_proba(X_new)[0, 1])

    return predicted_class, pass_probability


def format_hours(hours: float) -> str:
    return f"{hours:g} hour" if hours == 1 else f"{hours:g} hours"


def make_chart(model, study_hours, attendance_percentage,
               pass_probability, predicted_class):
    hours = np.linspace(0, MAX_HOURS, 100)

    curve = pd.DataFrame({
        "study_hours": hours,
        "attendance_percentage": [attendance_percentage] * len(hours),
    })

    curve["pass_probability"] = model.predict_proba(curve)[:, 1]

    line = (
        alt.Chart(curve)
        .mark_line(color="#4c78a8")
        .encode(
            x=alt.X("study_hours:Q", title="Study hours"),
            y=alt.Y(
                "pass_probability:Q",
                title="Probability of passing",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
        )
    )

    cutoff = (
        alt.Chart(pd.DataFrame({"y": [0.5]}))
        .mark_rule(strokeDash=[6, 4], color="gray")
        .encode(y="y:Q")
    )

    student = pd.DataFrame({
        "study_hours": [study_hours],
        "pass_probability": [pass_probability],
        "Predicted class": [
            PASS_LABEL if predicted_class == 1 else FAIL_LABEL
        ],
    })

    point = (
        alt.Chart(student)
        .mark_point(size=250, filled=True, opacity=1)
        .encode(
            x="study_hours:Q",
            y="pass_probability:Q",
            color=alt.Color(
                "Predicted class:N",
                scale=alt.Scale(
                    domain=[FAIL_LABEL, PASS_LABEL],
                    range=["#d62728", "#2ca02c"],
                ),
            ),
            tooltip=[
                "study_hours",
                alt.Tooltip("pass_probability:Q", format=".1%"),
                "Predicted class",
            ],
        )
    )

    return (line + cutoff + point).properties(height=320)


st.title("🎓 Student Pass Predictor")

st.write(
    "Enter a student's name, study hours, and attendance "
    "percentage to predict pass or fail."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file '{MODEL_PATH.name}' was not found. "
        "Run python train_model.py in this folder first."
    )
    st.stop()

model = load_model()

student_name = st.text_input(
    "Student name",
    placeholder="e.g. Maria Santos",
)

study_hours = st.number_input(
    "Study hours",
    min_value=0.0,
    max_value=float(MAX_HOURS),
    value=4.0,
    step=0.5,
)

attendance_percentage = st.number_input(
    "Attendance percentage",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=1.0,
)

if st.button("Predict"):
    name = student_name.strip()

    if not name:
        st.warning("Please enter the student's name first.")
    else:
        predicted_class, pass_probability = predict(
            model,
            study_hours,
            attendance_percentage,
        )

        if predicted_class == 1:
            st.success(
                f"{name} is predicted to PASS after "
                f"{format_hours(study_hours)} of study. "
                f"Pass probability: {pass_probability:.0%}."
            )
        else:
            st.error(
                f"{name} is predicted to FAIL after "
                f"{format_hours(study_hours)} of study. "
                f"Pass probability: {pass_probability:.0%}."
            )

        st.subheader("How was this prediction generated?")

        st.write(
            "The app uses a previously trained logistic regression "
            "model. It does not retrain the model during prediction."
        )

        st.write(f"Study hours: {study_hours:g}")
        st.write(f"Attendance: {attendance_percentage:.0f}%")
        st.write(f"Pass probability: {pass_probability:.1%}")

        st.subheader("Study hours vs. predicted probability")

        st.altair_chart(
            make_chart(
                model,
                study_hours,
                attendance_percentage,
                pass_probability,
                predicted_class,
            ),
            use_container_width=True,
        )

        st.caption(
            "The curve shows predicted pass probability at the "
            "selected attendance percentage. The dot represents "
            "this student's prediction. The dashed line marks 50%."
        )
