"""Exercise 3: Streamlit UI with study hours and attendance."""
from pathlib import Path
import joblib
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Prediction", page_icon="🎓")
st.title("Student Pass Prediction")
st.write("Enter the student's study hours and attendance percentage.")

if not MODEL_PATH.exists():
    st.warning("Model file not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
student_name = st.text_input("Student name", placeholder="Enter student name")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
)
attendance = st.number_input(
    "Attendance percentage (%)", min_value=0.0, max_value=100.0,
    value=80.0, step=1.0
)

if st.button("Predict result"):
    predicted = int(model.predict([[study_hours, attendance]])[0])
    result = "Pass" if predicted == 1 else "Fail"
    name = student_name.strip() or "The student"

    if predicted == 1:
        st.success(f"{name}: predicted result is {result}.")
    else:
        st.error(f"{name}: predicted result is {result}.")

    st.write(f"Study hours: **{study_hours}**")
    st.write(f"Attendance: **{attendance}%**")
   
    st.bar_chart({
        "Input values": [study_hours, attendance]
    }, x=None)
