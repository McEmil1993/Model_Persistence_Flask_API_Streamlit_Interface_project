
from pathlib import Path
import joblib
import streamlit as st
import matplotlib.pyplot as plt

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("Student Pass Predictor")
st.write("Enter a student's name and daily study hours to estimate a Pass/Fail result.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` in the project folder first.")
    st.stop()

model = joblib.load(MODEL_PATH)
student_name = st.text_input("Student name", placeholder="Enter student name")
study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0,
                               value=5.0, step=0.5)
if st.button("Predict result"):
    prediction = int(model.predict([[study_hours]])[0])
    label = "Pass" if prediction == 1 else "Fail"
    name = student_name.strip() or "Student"
    st.subheader(f"Prediction for {name}")
    if prediction == 1:
        st.success(f"Predicted class: {label}")
    else:
        st.warning(f"Predicted class: {label}")
    st.write(
        f"The model used {study_hours:g} study hour(s) per day as input. "
        "It was trained beforehand on example study-hours and Pass/Fail labels, "
        "then loaded from a saved file. This is an estimate, not a guaranteed outcome."
    )

    # Bonus chart: show the entered hours and the predicted class.
    fig, ax = plt.subplots()
    ax.bar(["Study hours", "Predicted class (0=Fail, 1=Pass)"],
           [study_hours, prediction])
    ax.set_title("Study Hours and Predicted Class")
    ax.set_ylabel("Value")
    st.pyplot(fig)
