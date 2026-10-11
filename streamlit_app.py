
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).resolve().parent / "studymate_model.joblib"

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🌷",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #08080d 0%, #171220 55%, #21162b 100%);
    color: #f5efff;
}

.block-container {
    max-width: 850px;
    padding-top: 2.2rem;
}

h1, h2, h3, p, label,
[data-testid="stMarkdownContainer"] {
    color: #f5efff;
}

div[data-testid="stMetric"] {
    background: #1c1725;
    border: 1px solid #59416f;
    padding: 16px;
    border-radius: 16px;
}

div[data-testid="stMetric"] label {
    color: #cdb8e8;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

.stButton > button {
    background: #a16bd4;
    color: #ffffff;
    border: 1px solid #c49bea;
    border-radius: 12px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #bd8af0;
    color: #171220;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    color: #ffffff !important;
    background-color: #262230 !important;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stAlert"] p {
    color: #ffffff !important;
}

[data-testid="stCaptionContainer"] p {
    color: #cdb8e8;
}
</style>
""", unsafe_allow_html=True)

st.title("🌷 StudyMate AI")
st.write("A simple machine-learning study-support demonstration.")
st.caption("Explore how study time and attendance relate to a sample learning-status prediction.")

if not MODEL_FILE.exists():
    st.error("The saved model is missing. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_FILE)

with st.container(border=True):
    st.subheader("Student details")

    student = st.text_input(
        "Student name",
        placeholder="Enter a name"
    )

    col1, col2 = st.columns(2)

    with col1:
        hours = st.number_input(
            "Study hours per day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

    with col2:
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0
        )

    run_prediction = st.button(
        "Check study status",
        use_container_width=True
    )

if run_prediction:
    if not student.strip():
        st.warning("Please enter the student's name.")
    else:
        label = str(model.predict([[float(hours), float(attendance)]])[0])

        st.subheader("Study summary")
        st.write(f"**Student:** {student.strip()}")

        left, right = st.columns(2)
        left.metric("Study hours/day", f"{hours:g}")
        right.metric("Attendance", f"{attendance:g}%")

        if label == "On Track":
            st.success(f"Sample prediction: {label}")
        else:
            st.warning(f"Sample prediction: {label}")

        st.info(
            "This result comes from a small, synthetic classroom dataset. "
            "It demonstrates model persistence and prediction; it is not a "
            "reliable evaluation of a real student's ability."
        )

        st.subheader("Your inputs")
        st.write("The bars show two different measurements, each with its own unit.")

        chart_data = pd.DataFrame(
            {"Value": [float(hours), float(attendance)]},
            index=["Study hours per day", "Attendance (%)"]
        )

        st.bar_chart(chart_data, horizontal=True)
        st.caption(f"Model output: {label} (demonstration label, not a grade).")

st.divider()
st.caption("StudyMate AI · IT Elective 3 — Machine Learning")
