# Student Pass Prediction Interface - Pastel UI
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(
    page_title="Student Success Studio",
    page_icon="🌷",
    layout="wide",
)

# Pastel styling only: prediction logic and model inputs remain unchanged.
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fffaf5 0%, #f7f0ff 55%, #eef8ff 100%);
        color: #34283e;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, label { color: #34283e; }
    .hero-card {
        background: linear-gradient(115deg, #fce7f3 0%, #f3e8ff 58%, #e8f4ff 100%);
        border: 1px solid #ead9f0;
        border-radius: 26px;
        padding: 1.7rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(91, 58, 111, 0.07);
    }
    .hero-eyebrow {
        color: #85518f;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }
    .hero-title {
        color: #34283e;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.1;
        font-weight: 800;
        margin: 0;
    }
    .hero-subtitle {
        color: #65566e;
        font-size: 1.05rem;
        margin-top: .7rem;
        margin-bottom: 0;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, .82);
        border: 1px solid #eadfed;
        border-radius: 20px;
        box-shadow: 0 7px 20px rgba(72, 48, 87, .045);
    }
    .stTextInput input, .stNumberInput input {
        background: #fff;
        border-radius: 11px;
    }
    .stButton > button {
        background: #85519a;
        color: #fff;
        border: 1px solid #85519a;
        border-radius: 12px;
        min-height: 2.8rem;
        font-weight: 700;
        transition: all .15s ease-in-out;
    }
    .stButton > button:hover {
        background: #6f3d84;
        border-color: #6f3d84;
        color: #fff;
    }
    [data-testid="stMetric"] {
        background: #fff7fc;
        border: 1px solid #f0dceb;
        border-radius: 15px;
        padding: .85rem 1rem;
    }
    [data-testid="stCaptionContainer"] p { color: #74667b; }
    hr { border-color: #eadfed; }
    </style>
    <div class="hero-card">
        <div class="hero-eyebrow">Academic Support · Machine Learning Demo</div>
        <p class="hero-title">Student Success Studio 🌷</p>
        <p class="hero-subtitle">
            Explore how study time and attendance affect a sample pass/fail prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first, then restart this app.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Same inputs and prediction workflow as the original version, with a new layout.
with st.container(border=True):
    st.subheader("✏️ Student Details")
    st.write("Enter the student's information to generate a prediction.")
    input_left, input_right = st.columns([1.15, 1])

    with input_left:
        student_name = st.text_input("Student name", placeholder="e.g., Alex Rivera")
        study_hours = st.number_input(
            "Study hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
        )

    with input_right:
        attendance_percentage = st.slider(
            "Attendance percentage (%)",
            min_value=0,
            max_value=100,
            value=80,
            step=1,
        )
        st.caption("Choose attendance from 0% to 100%.")
        st.markdown("**Inputs used:** study hours + attendance")

    predict_clicked = st.button("🌷 Predict Result", use_container_width=True)

if predict_clicked:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        # Keep feature order consistent with train_model.py: [study_hours, attendance_percentage]
        input_values = [[float(study_hours), float(attendance_percentage)]]
        prediction_value = int(model.predict(input_values)[0])
        result = "Pass" if prediction_value == 1 else "Fail"

        st.subheader("🌟 Prediction Summary")
        with st.container(border=True):
            metric_left, metric_middle, metric_right = st.columns(3)
            metric_left.metric("Predicted result", result)
            metric_middle.metric("Study hours", f"{study_hours:g} hrs")
            metric_right.metric("Attendance", f"{attendance_percentage}%")

            if result == "Pass":
                st.success(f"{student_name.strip()} is predicted to Pass.")
            else:
                st.error(f"{student_name.strip()} is predicted to Fail.")

            st.write(
                f"**How it was generated:** The app loaded the saved machine-learning model "
                f"and used both inputs: {study_hours:g} study hours and "
                f"{attendance_percentage}% attendance."
            )
            st.caption(
                "This project uses illustrative sample training data, not real student records. "
                "Predictions are for demonstrating the exercise and should not be treated as "
                "reliable academic assessments."
            )

        st.subheader("📊 Explore the Inputs")
        chart_left, chart_right = st.columns(2)

        # Bonus chart: study hours vs. predicted class with attendance held constant.
        hours_to_compare = sorted(
            {float(hours) for hours in range(0, 25)} | {float(study_hours)}
        )
        hour_inputs = [
            [hours, float(attendance_percentage)] for hours in hours_to_compare
        ]
        hour_predictions = model.predict(hour_inputs)
        hours_chart = pd.DataFrame({
            "Study hours": hours_to_compare,
            "Predicted class (0 = Fail, 1 = Pass)": [
                int(value) for value in hour_predictions
            ],
        })

        with chart_left:
            with st.container(border=True):
                st.markdown("#### 📚 Study Hours vs. Predicted Class")
                st.caption(
                    f"Attendance is held constant at {attendance_percentage}%. "
                    "0 = Fail, 1 = Pass."
                )
                st.line_chart(hours_chart.set_index("Study hours"), height=280)

        # Additional chart: attendance vs. predicted class with study hours held constant.
        attendance_values = list(range(0, 101, 5))
        attendance_inputs = [
            [float(study_hours), attendance]
            for attendance in attendance_values
        ]
        attendance_predictions = model.predict(attendance_inputs)
        attendance_chart = pd.DataFrame({
            "Attendance percentage": attendance_values,
            "Predicted class (0 = Fail, 1 = Pass)": [
                int(value) for value in attendance_predictions
            ],
        })

        with chart_right:
            with st.container(border=True):
                st.markdown("#### 🗓️ Attendance vs. Predicted Class")
                st.caption(
                    f"Study hours are held constant at {study_hours:g}. "
                    "0 = Fail, 1 = Pass."
                )
                st.line_chart(
                    attendance_chart.set_index("Attendance percentage"),
                    height=280,
                )

st.divider()
st.caption("Student Success Studio · Educational machine-learning demonstration")
