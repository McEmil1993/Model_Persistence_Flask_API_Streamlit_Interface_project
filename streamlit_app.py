"""
Exercise 3 — Streamlit Interface
Run with: streamlit run streamlit_app.py
"""
from pathlib import Path
import joblib
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Pass Predictor")
st.write("Enter a student's name and study hours to get a practice prediction.")

if not MODEL_PATH.exists():
    st.error("The model file is missing. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

student_name = st.text_input("Student name", placeholder="Enter student name")
study_hours = st.number_input(
    "Study hours",
    min_value=0.0,
    max_value=24.0,
    value=3.0,
    step=0.5,
    help="Number of hours studied for the relevant period."
)

if st.button("Predict result", type="primary"):
    prediction = int(model.predict([[study_hours]])[0])
    result = "Pass" if prediction == 1 else "Fail"
    display_name = student_name.strip() or "The student"

    if prediction == 1:
        st.success(f"{display_name} is predicted to **Pass**.")
    else:
        st.warning(f"{display_name} is predicted to **Fail**.")

    st.caption(f"Study hours entered: {study_hours:g}")
    st.info(
        "How this prediction was generated: the app loaded the saved "
        "`student_pass_model.joblib` model and passed the study-hours value "
        "to its `predict()` method. This practice model was trained on example "
        "data and is not a guarantee of actual academic performance."
    )

    # Bonus chart: show entered study hours and predicted class.
    fig, ax = plt.subplots()
    ax.bar(["Study hours", "Predicted class"], [study_hours, prediction])
    ax.set_title("Input and prediction")
    ax.set_ylabel("Value (hours / class 0 or 1)")
    ax.set_ylim(0, max(1, study_hours) + 1)
    st.pyplot(fig)
    plt.close(fig)
