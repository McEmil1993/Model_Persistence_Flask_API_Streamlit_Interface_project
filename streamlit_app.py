
"""
Exercise 3: Student Pass/Fail Predictor

Uses study hours and attendance percentage to predict a result
with the saved model from train_model.py.

The training dataset is simulated for educational purposes.
Predictions are not guarantees of real student performance.
"""

from pathlib import Path

import joblib
import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="🎓",
    layout="centered",
)

# --------------------------------------------------
# Custom design
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #eef0f3;
        color: #111827;
    }

    .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #111111 !important;
        font-weight: 750 !important;
    }

    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: #222222 !important;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label {
        color: #111111 !important;
        font-weight: 650 !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #9ca3af !important;
        border-radius: 8px !important;
        min-height: 45px;
    }

    div.stButton > button {
        background-color: #d1d5db !important;
        color: #111111 !important;
        border: 1px solid #9ca3af !important;
        border-radius: 9px !important;
        min-height: 46px;
        padding: 0.65rem 1.3rem;
        font-weight: 700 !important;
    }

    div.stButton > button:hover {
        background-color: #bfc5ce !important;
        color: #000000 !important;
        border-color: #6b7280 !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #c5c9d0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Heading
# --------------------------------------------------

st.title("🎓 Student Pass/Fail Predictor")

st.markdown(
    "<p style='text-align:center;'>"
    "Estimate a student's result using study hours "
    "and attendance percentage."
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Make sure "
        "student_pass_model.joblib is in the same folder "
        "as this file, then run: python train_model.py"
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

# --------------------------------------------------
# Student inputs
# --------------------------------------------------

st.subheader("Student Information")
st.write("Enter the student's details below.")

with st.form("student_prediction_form"):
    student_name = st.text_input(
        "Student name",
        placeholder="Enter student name",
    )

    study_hours = st.number_input(
        "Study hours (0–24)",
        min_value=0.0,
        max_value=24.0,
        value=0.0,
        step=0.5,
        format="%.2f",
        help="Enter the number of hours spent studying per day.",
    )

    attendance = st.number_input(
        "Attendance",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        format="%.2f",
        help="Enter a number from 0 to 100. For example, enter 85 for 85% attendance.",
    )

    # Display the percentage automatically.
    st.caption(f"Attendance entered: {attendance:g}%")

    submitted = st.form_submit_button(
        "Predict Result",
        use_container_width=True,
    )

# --------------------------------------------------
# Generate prediction
# --------------------------------------------------

if submitted:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    elif not (
        math.isfinite(float(study_hours))
        and math.isfinite(float(attendance))
    ):
        st.warning("Please enter valid numerical values.")
    else:
        # Feature order must match train_model.py:
        # StudyHours, then Attendance.
        prediction = str(
            model.predict([
                [float(study_hours), float(attendance)]
            ])[0]
        )

        st.session_state["last_student_name"] = student_name.strip()
        st.session_state["last_study_hours"] = float(study_hours)
        st.session_state["last_attendance"] = float(attendance)
        st.session_state["last_prediction"] = prediction

# --------------------------------------------------
# Display most recent prediction
# --------------------------------------------------

if "last_prediction" in st.session_state:
    last_name = st.session_state["last_student_name"]
    last_hours = st.session_state["last_study_hours"]
    last_attendance = st.session_state["last_attendance"]
    last_prediction = st.session_state["last_prediction"]

    st.divider()
    st.subheader(f"Prediction for {last_name}")

    if last_prediction.lower() == "pass":
        st.success(f"Predicted Result: {last_prediction}")
    else:
        st.warning(f"Predicted Result: {last_prediction}")

    st.markdown("### How was this prediction generated?")

    st.write(
        f"The model received **{last_hours:g} study hours** "
        f"and **{last_attendance:g}% attendance**. "
        "It processed both values using the patterns learned "
        "from the simulated training dataset to predict the result."
    )

    st.caption(
        "This is an estimate, not a guarantee of actual academic "
        "performance. The model was trained on simulated data, "
        "so its evaluation scores do not establish real-world accuracy."
    )

    # --------------------------------------------------
    # Chart: study hours versus predicted result
    # Attendance is held at the student's entered value.
    # --------------------------------------------------

    st.divider()
    st.markdown("### Study Hours vs. Predicted Result")

    st.write(
        f"This chart shows the model's predictions from 0 to 24 "
        f"study hours while keeping attendance fixed at "
        f"{last_attendance:g}%. The dashed line marks "
        f"student's entered study hours ({last_hours:g} hours), "
        f"and the marker shows the predicted result at that point."
    )

    chart_hours = np.linspace(0, 24, 97)

    chart_inputs = pd.DataFrame({
        "StudyHours": chart_hours,
        "Attendance": [last_attendance] * len(chart_hours),
    })

    chart_predictions = model.predict(chart_inputs)

    chart_data = pd.DataFrame({
        "Study Hours": chart_hours,
        "Predicted Result": [str(value) for value in chart_predictions],
    })

    # Map the predicted classes to two chart positions.
    chart_data["Predicted Class"] = chart_data[
        "Predicted Result"
    ].map({
        "Fail": 0,
        "Pass": 1,
    })

    chart = (
        alt.Chart(chart_data)
        .mark_line(point=alt.OverlayMarkDef(size=35))
        .encode(
            x=alt.X(
                "Study Hours:Q",
                scale=alt.Scale(domain=[0, 24]),
                axis=alt.Axis(title="Study Hours", tickMinStep=2),
            ),
            y=alt.Y(
                "Predicted Class:Q",
                scale=alt.Scale(domain=[-0.1, 1.1]),
                axis=alt.Axis(
                    title="Model Prediction",
                    values=[0, 1],
                    labelExpr="datum.value == 0 ? 'Fail' : 'Pass'",
                ),
            ),
            tooltip=[
                alt.Tooltip("Study Hours:Q", format=".2f"),
                alt.Tooltip("Predicted Result:N"),
            ],
        )
    )

    selected_point = pd.DataFrame({
        "Study Hours": [last_hours],
        "Predicted Result": [last_prediction],
        "Predicted Class": [
            1 if last_prediction.lower() == "pass" else 0
        ],
    })

    selected_marker = (
        alt.Chart(selected_point)
        .mark_point(
            filled=True,
            size=180,
            color="#d97706",
            stroke="#111827",
            strokeWidth=1.5,
        )
        .encode(
            x=alt.X(
                "Study Hours:Q",
                scale=alt.Scale(domain=[0, 24]),
            ),
            y=alt.Y(
                "Predicted Class:Q",
                scale=alt.Scale(domain=[-0.1, 1.1]),
            ),
            tooltip=[
                alt.Tooltip("Study Hours:Q", format=".2f"),
                alt.Tooltip("Predicted Result:N"),
            ],
        )
    )

    selected_rule = (
        alt.Chart(selected_point)
        .mark_rule(
            color="#555555",
            strokeDash=[6, 4],
            strokeWidth=2,
        )
        .encode(
            x=alt.X(
                "Study Hours:Q",
                scale=alt.Scale(domain=[0, 24]),
            )
        )
    )

    st.altair_chart(
        (chart + selected_rule + selected_marker).properties(
            height=320
        ),
        use_container_width=True,
    )

    st.caption(
        "The orange marker identifies this student's study hours "
        "and predicted class. The chart reflects the trained model, "
        "not a guaranteed academic outcome."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.markdown(
    "<p style='text-align:center; font-size:12px;'>"
    "Student Result Predictor | Machine Learning Project"
    "</p>",
    unsafe_allow_html=True,
)