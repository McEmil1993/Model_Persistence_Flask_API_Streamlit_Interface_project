"""
Exercise 3 — Streamlit Interface
Run: streamlit run streamlit_app.py
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Pass Predictor")
st.write("Estimate a pass/fail class from a student's daily study hours.")

if not MODEL_PATH.exists():
    st.error("The trained model file was not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

student_name = st.text_input("Student name", placeholder="Enter the student's name")
study_hours = st.number_input(
    "Study hours per day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
)

if st.button("Predict result", type="primary", use_container_width=True):
    prediction = int(model.predict([[study_hours]])[0])
    result = "PASS" if prediction == 1 else "FAIL"
    display_name = student_name.strip() or "The student"

    if prediction == 1:
        st.success(f"{display_name} is predicted to: {result}")
    else:
        st.warning(f"{display_name} is predicted to: {result}")

    st.caption(
        f"Prediction generated from {study_hours:g} study hour(s) per day using "
        "the previously trained Logistic Regression model. This is an educational "
        "demo, not a guarantee of academic performance."
    )

    # Bonus chart: compare several study-hour values with predicted classes.
    hours_range = np.arange(0, 13, 1, dtype=float)
    classes = model.predict(hours_range.reshape(-1, 1))
    fig, ax = plt.subplots()
    ax.step(hours_range, classes, where="mid", marker="o")
    ax.set_xlabel("Study hours per day")
    ax.set_ylabel("Predicted class (0 = Fail, 1 = Pass)")
    ax.set_yticks([0, 1], labels=["Fail (0)", "Pass (1)"])
    ax.set_xticks(hours_range)
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

st.info(
    "Model limitation: this version uses study hours only. The name is for display "
    "only, and attendance is not part of this Streamlit model. To use more inputs, "
    "update the training data and retrain a model with those features."
)
