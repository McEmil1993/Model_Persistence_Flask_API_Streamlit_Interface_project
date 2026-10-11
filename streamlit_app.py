
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Student Prediction System",
    page_icon="🎓",
    layout="centered"
)

# Model location
MODEL_PATH = Path(__file__).with_name(
    "student_pass_model.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Page design
st.title("🎓 Student Prediction System")
st.write(
    "Predict a student's likely academic result "
    "based on study hours and attendance."
)

st.info(
    "This is an educational demonstration using "
    "synthetic sample data."
)

# Check whether the trained model exists
if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. "
        "Run: python train_model.py"
    )
    st.stop()

model = load_model()

# Input form
st.subheader("Student Information")

with st.form("prediction_form"):
    name = st.text_input("Student Name")

    study_hours = st.number_input(
        "Study Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )

    submitted = st.form_submit_button(
        "Predict Result",
        use_container_width=True
    )

# Generate prediction
if submitted:
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        student = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance_percentage": attendance
        }])

        prediction = model.predict(student)[0]
        probabilities = model.predict_proba(student)[0]
        classes = list(model.classes_)

        pass_probability = float(
            probabilities[classes.index("PASS")]
        )

        st.divider()
        st.subheader("Prediction Result")
        st.write(f"Student: **{name.strip()}**")

        if prediction == "PASS":
            st.success("Predicted Result: PASS")
        else:
            st.error("Predicted Result: FAIL")

        st.metric(
            "Estimated Pass Probability",
            f"{pass_probability:.2%}"
        )

        st.progress(pass_probability)

        st.caption(
            "The prediction is based on synthetic data "
            "and is not a reliable assessment of actual "
            "academic performance."
        )
