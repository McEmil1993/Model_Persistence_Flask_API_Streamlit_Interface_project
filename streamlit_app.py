"""
EXERCISE 3 — STREAMLIT INTERFACE

Features:
- Student name input
- Study-hours input
- Predict button
- Pass/Fail result message
- Short explanation of how the prediction was generated
- Bonus chart comparing study hours with the predicted class

Install the required packages:
    pip install streamlit scikit-learn joblib pandas

Run this file:
    streamlit run exercise3_streamlit_app.py

Note: This is a classroom demonstration using sample data. The model predicts
using study_hours only. It is not a real assessment of academic performance.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"


def train_and_save_model():
    """Train and save a sample Pass/Fail model if one is not already available."""
    study_hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    results = [
        "Fail", "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(study_hours, results)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_resource
def load_model():
    """Load the saved model or create it if the file does not exist."""
    if not MODEL_PATH.exists():
        train_and_save_model()
    return joblib.load(MODEL_PATH)


def predict_result(model, hours):
    if hours < 0:
        raise ValueError("Study hours cannot be negative.")
    return str(model.predict([[hours]])[0])


st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="📘",
    layout="centered"
)

st.title("Student Pass/Fail Predictor")
st.write(
    "Enter the student's name and study hours, then click **Predict** "
    "to see the model's prediction."
)

model = load_model()

student_name = st.text_input(
    "Student name",
    placeholder="Enter student name"
)

study_hours = st.number_input(
    "Study hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

if st.button("Predict", type="primary"):
    prediction = predict_result(model, study_hours)
    name = student_name.strip() or "The student"

    if prediction == "Pass":
        st.success(f"{name}: Predicted result — PASS")
    else:
        st.error(f"{name}: Predicted result — FAIL")

    st.write(
        f"The model used **{study_hours:g} study hours** as input. "
        "It compared this input with patterns learned from the sample "
        "training data to predict Pass or Fail."
    )

    st.caption(
        "This is a classroom demonstration using sample data, not a real "
        "assessment of a student's academic performance."
    )

st.divider()
st.subheader("Bonus: Study Hours vs. Predicted Class")

hours_range = list(range(0, 11))
predicted_classes = [predict_result(model, hours) for hours in hours_range]

chart_data = pd.DataFrame({
    "Study hours": hours_range,
    "Predicted class (Pass = 1, Fail = 0)": [
        1 if result == "Pass" else 0 for result in predicted_classes
    ]
}).set_index("Study hours")

st.line_chart(chart_data)
st.caption("In the chart, 1 represents Pass and 0 represents Fail.")
