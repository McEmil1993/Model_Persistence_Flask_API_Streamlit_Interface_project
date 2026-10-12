# Exercise 3 — Streamlit Interface

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Result Predictor", page_icon="🎓")
st.title("Student Result Predictor")
st.write("Estimate a student's result using their study hours.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

student_name = st.text_input("Student name")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
)

if st.button("Predict result"):
    prediction = str(model.predict([[study_hours]])[0])
    display_name = student_name.strip() or "The student"

    if prediction == "Pass":
        st.success(f"{display_name} is predicted to: {prediction}")
    else:
        st.warning(f"{display_name} is predicted to: {prediction}")

    st.write(
        f"**How it was generated:** The saved machine-learning model received "
        f"{study_hours:g} study hours as its input and returned **{prediction}**. "
        "This is an estimate from the example training data, not a guarantee."
    )

    chart_data = pd.DataFrame({
        "Study hours": [study_hours],
        "Predicted class (1 = Pass, 0 = Fail)": [1 if prediction == "Pass" else 0]
    })
    st.subheader("Study hours and predicted class")
    st.bar_chart(chart_data.set_index("Study hours"))

st.caption(
    "The example model currently uses study hours only. Attendance is not used "
    "by this Streamlit model unless the model is retrained with attendance data."
)
