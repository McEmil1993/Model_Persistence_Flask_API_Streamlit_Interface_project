from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model_v2.joblib"

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}. Run 'py train_model_v2.py' first.")
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🎓 Student Pass Predictor")
st.write("Enter a student's details to predict whether they are likely to pass.")

student_name = st.text_input("Student name")
study_hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
attendance_percentage = st.slider("Attendance percentage", min_value=0, max_value=100, value=75)

if st.button("Predict"):
    if model is None:
        st.stop()
    if not student_name.strip():
        st.warning("Please enter the student's name first.")
    else:
        sample = pd.DataFrame({
            "study_hours": [study_hours],
            "attendance_percentage": [attendance_percentage],
        })
        prediction = int(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0][1])

        if prediction == 1:
            st.success(
                f"✅ {student_name} is predicted to PASS "
                f"(pass probability: {probability:.0%})."
            )
        else:
            st.error(
                f"❌ {student_name} is predicted to FAIL "
                f"(pass probability: {probability:.0%})."
            )

        coef_hours = float(model.coef_[0][0])
        coef_attendance = float(model.coef_[0][1])
        intercept = float(model.intercept_[0])
        st.subheader("How was this prediction generated?")
        st.write(
            f"The saved model is a logistic regression trained on study hours and attendance percentage. "
            f"It loaded its learned values and applied them to the {study_hours} hours and {attendance_percentage}% attendance you entered."
        )
        st.latex(r"P(\text{pass}) = \frac{1}{1 + e^{-(b_0 + b_1 \cdot \text{hours} + b_2 \cdot \text{attendance})}}")
        st.write(
            f"With the learned values b₀ = {intercept:.2f}, b₁ = {coef_hours:.2f}, and b₂ = {coef_attendance:.2f}, "
            f"the pass probability is {probability:.2f}. If it is above 0.5, the model predicts PASS; otherwise it predicts FAIL."
        )

        st.subheader("Study hours vs predicted class")
        hours_range = np.linspace(0, 12, 200)
        curve = model.predict_proba(
            pd.DataFrame({
                "study_hours": hours_range,
                "attendance_percentage": [attendance_percentage] * len(hours_range),
            })
        )[:, 1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hours_range, curve, label="Pass probability curve")
        ax.axhline(0.5, color="gray", linestyle="--", label="PASS / FAIL threshold (0.5)")
        ax.scatter(
            [study_hours],
            [probability],
            color="green" if prediction == 1 else "red",
            s=120,
            zorder=5,
            label=f"{student_name}: {'PASS' if prediction == 1 else 'FAIL'}",
        )
        ax.set_xlabel("Study hours")
        ax.set_ylabel("Pass probability")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        st.pyplot(fig)