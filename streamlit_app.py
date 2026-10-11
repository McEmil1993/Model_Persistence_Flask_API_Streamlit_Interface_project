"""
Exercise 3 — Streamlit Interface
Run with: streamlit run streamlit_app.py
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Prediction", page_icon="🎓")
st.title("🎓 Student Pass Prediction")
st.write("Enter a student's name and study hours to estimate a Pass/Fail class.")

if not MODEL_PATH.exists():
    st.error("Model file not found. In this folder, run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    student_name = st.text_input("Student name", placeholder="Enter student name")
    study_hours = st.number_input(
        "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    if not student_name.strip():
        st.warning("Please enter a student name.")
    else:
        predicted_class = int(model.predict([[float(study_hours)]])[0])
        result = "Pass" if predicted_class == 1 else "Fail"
        st.subheader("Prediction result")
        st.success(f"{student_name.strip()}: predicted class — {result}")

        st.write(
            f"The model received **{study_hours:g} study hours** and returned "
            f"the class **{result}**. It was trained on example study-hour data "
            "and learned decision rules from those examples."
        )
        st.caption(
            "This is a classroom demonstration, not a reliable assessment of a real student. "
            "The model uses study hours only; attendance is not included."
        )

        # Bonus chart: compare the entered study hours with the predicted class.
        fig, ax = plt.subplots()
        ax.bar(["Study hours", "Predicted class (0=Fail, 1=Pass)"],
               [float(study_hours), predicted_class])
        ax.set_ylabel("Value")
        ax.set_title(f"Study hours and predicted class for {student_name.strip()}")
        st.pyplot(fig)
        plt.close(fig)
