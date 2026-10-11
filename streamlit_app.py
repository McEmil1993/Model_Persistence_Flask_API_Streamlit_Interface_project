"""
Exercise 3 — Streamlit Interface
Run: streamlit run streamlit_app.py
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")
st.write("Enter a student's study hours to estimate whether they may pass.")

if not MODEL_FILE.exists():
    st.error("The model file is missing. Open the VS Code terminal and run: python train_model.py")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(MODEL_FILE)

model = get_model()

with st.form("student_prediction_form"):
    student_name = st.text_input("Student name", placeholder="Enter student name")
    study_hours = st.number_input(
        "Study hours per day", min_value=0.0, max_value=24.0,
        value=5.0, step=0.5
    )
    submitted = st.form_submit_button("Predict result")

if submitted:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        sample = pd.DataFrame([{"study_hours": float(study_hours)}])
        predicted_class = int(model.predict(sample)[0])
        result = "PASS" if predicted_class == 1 else "FAIL"

        st.subheader(f"Prediction for {student_name.strip()}")
        if predicted_class == 1:
            st.success("Estimated result: PASS")
        else:
            st.error("Estimated result: FAIL")

        st.write(f"**Study hours per day:** {study_hours}")
        st.info(
            "Explanation: The model uses study hours to estimate the result. "
            "This is a prediction based on the sample training "
            "dataset, not a guarantee of the student's actual grade."
        )

        chart_data = pd.DataFrame({
            "Input": ["Study hours per day"],
            "Value": [study_hours]
        }).set_index("Input")
        st.bar_chart(chart_data)