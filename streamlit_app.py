"""Simple Streamlit interface for the student pass classifier."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("Student Pass Predictor")
st.write("Estimate a pass or fail class using study hours. Attendance is collected for reference but does not affect this model's prediction.")

if not MODEL_PATH.exists():
    st.error("The model file is missing. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    student_name = st.text_input("Student name")
    study_hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    attendance_percentage = st.number_input(
        "Attendance percentage",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0,
        help="Must be between 0 and 100. The current model does not use attendance to calculate its prediction.",
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    if not student_name.strip():
        st.warning("Enter a student name to continue.")
    else:
        predicted_class = int(model.predict([[study_hours]])[0])
        label = "Pass" if predicted_class == 1 else "Fail"
        st.success(f"{student_name.strip()} is predicted to: {label}")
        st.metric("Attendance entered", f"{attendance_percentage:g}%")
        st.caption(
            f"Prediction inputs: {study_hours:g} study hours and {attendance_percentage:g}% attendance. "
            "The saved model uses study hours and the training labels only; attendance is collected "
            "and shown here but does not affect the prediction."
        )

        st.subheader("Input and prediction chart")
        chart_data = pd.DataFrame(
            {
                "Study hours": [study_hours],
                "Predicted class": [predicted_class],
                "Outcome": [label],
            }
        )
        st.scatter_chart(
            chart_data,
            x="Study hours",
            y="Predicted class",
            color="Outcome",
            size=200,
        )
        st.caption("Each point shows the study hours against the predicted class (0 = Fail, 1 = Pass).")
