"""Interactive Streamlit interface for the saved student pass/fail model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Pass/Fail Predictor")
st.write(
    "Enter a student's name and average daily study hours to get a prediction "
    "from the saved machine-learning model."
)

if not MODEL_PATH.exists():
    st.error("The model file is missing. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    student_name = st.text_input("Student name", placeholder="e.g., Alex")
    study_hours = st.number_input(
        "Average study hours per day",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5
    )
    submitted = st.form_submit_button("Predict result")

if submitted:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = int(model.predict([[float(study_hours)]])[0])
        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction")
        st.write(f"**Student:** {student_name.strip()}")
        if result == "Pass":
            st.success(f"Predicted result: {result}")
        else:
            st.warning(f"Predicted result: {result}")

        st.info(
            "This is an educational demonstration using a small sample dataset. "
            "The prediction is based on study hours only and is not a guarantee "
            "of actual academic performance."
        )

st.divider()
st.subheader("Sample data visualization")
chart_data = pd.DataFrame({
    "Study hours": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8]
})
chart_data["Predicted class"] = [
    "Pass" if int(model.predict([[hours]])[0]) == 1 else "Fail"
    for hours in chart_data["Study hours"]
]
st.dataframe(chart_data, use_container_width=True, hide_index=True)
st.caption(
    "The sample data and model are for learning only. Use a representative, "
    "properly evaluated dataset for real-world decisions."
)
