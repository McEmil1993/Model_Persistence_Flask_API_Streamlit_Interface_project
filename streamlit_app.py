"""
Exercise 3 — Streamlit Interface
Run: streamlit run streamlit_app.py
"""
from pathlib import Path
import joblib
import streamlit as st
import matplotlib.pyplot as plt

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Pass Predictor")
st.write("Estimate a student's result based on study hours.")

if not MODEL_PATH.exists():
    st.warning("The model file has not been created yet. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

student_name = st.text_input("Student name", placeholder="Enter student name")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
)

if st.button("Predict result", type="primary"):
    prediction = int(model.predict([[study_hours]])[0])
    result = "Pass" if prediction == 1 else "Fail"

    if student_name.strip():
        st.subheader(f"Prediction for {student_name.strip()}")
    else:
        st.subheader("Prediction result")

    if prediction == 1:
        st.success(f"Predicted class: {result}")
    else:
        st.error(f"Predicted class: {result}")

    st.info(
        f"The model received {study_hours:g} study hour(s) and used patterns "
        "learned from the training examples to predict the class. This is an "
        "estimate, not a guarantee of the student's actual result."
    )

    # Bonus chart: show this student's study hours and predicted class.
    fig, ax = plt.subplots()
    ax.bar(["Study hours", "Predicted class (0=Fail, 1=Pass)"],
           [study_hours, prediction])
    ax.set_title("Study Hours and Predicted Class")
    ax.set_ylabel("Value")
    st.pyplot(fig)
    plt.close(fig)
