"""
Simple Streamlit interface for student pass/fail predictions.
Run: streamlit run streamlit_app.py
"""
from pathlib import Path
import joblib
import streamlit as st
import matplotlib.pyplot as plt

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass/Fail Predictor")
st.write("Enter the student's details and click Predict.")

if not MODEL_PATH.exists():
    st.error("Model file not found. In the terminal, run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

student_name = st.text_input("Student name", placeholder="Enter student name")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
)
attendance = st.number_input(
    "Attendance percentage (%)",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=1.0
)

if st.button("Predict result"):
    prediction_value = int(model.predict([[study_hours]])[0])
    prediction = "Pass" if prediction_value == 1 else "Fail"
    display_name = student_name.strip() or "The student"

    if prediction == "Pass":
        st.success(f"{display_name} is predicted to **Pass**.")
    else:
        st.warning(f"{display_name} is predicted to **Fail**.")

    st.write(f"Study hours: **{study_hours}**")
    st.write(f"Attendance: **{attendance}%**")
    st.caption(
        "Explanation: the saved Decision Tree model was loaded from "
        "student_pass_model.joblib and used to predict from study hours. "
        "Attendance is collected for this exercise but is not used by this model."
    )

    # Bonus chart: show the model's predicted class for several study-hour values.
    hours_range = list(range(0, 9))
    predicted_classes = model.predict([[h] for h in hours_range])
    labels = ["Fail" if value == 0 else "Pass" for value in predicted_classes]

    fig, ax = plt.subplots()
    ax.bar([str(h) for h in hours_range], [int(v) for v in predicted_classes])
    ax.set_xlabel("Study hours")
    ax.set_ylabel("Predicted class (0 = Fail, 1 = Pass)")
    ax.set_title("Study Hours vs Predicted Class")
    ax.set_yticks([0, 1], labels=["Fail", "Pass"])
    st.pyplot(fig)
