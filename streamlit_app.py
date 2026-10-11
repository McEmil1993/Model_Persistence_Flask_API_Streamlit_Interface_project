
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path("student_pass_model.joblib")

st.title("Student Pass Prediction")
st.subheader("Machine Learning Model Demo")

if not MODEL_PATH.exists():
    st.warning("Model file not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

name = st.text_input("Student Name")
hours = st.number_input("Study Hours", min_value=0.0, step=0.5)
attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict"):
    if not name:
        st.warning("Please enter the student's name.")
        st.stop()

    if attendance < 0 or attendance > 100:
        st.error("Attendance must be between 0 and 100.")
        st.stop()

    result = model.predict([[hours]])
    prediction = "Pass" if result[0] == 1 else "Fail"

    if prediction == "Pass":
        st.success(f"{name} is likely to Pass.")
    else:
        st.error(f"{name} is likely to Fail.")

    st.write(f"This prediction was generated using the trained model with the entered study hours: {hours} hours.")
    st.write(f"Student Name: {name}")
    st.write(f"Study Hours: {hours}")
    st.write(f"Attendance: {attendance}%")
    st.write(f"Prediction: {prediction}")

    st.bar_chart({"Study Hours": [hours], "Prediction": [result[0]]})
