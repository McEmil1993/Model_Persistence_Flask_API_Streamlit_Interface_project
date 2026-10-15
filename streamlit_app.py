"""
Exercise 3 — Streamlit Interface

Interactive web UI for the student pass/fail predictor.
Features:
  • Student name input
  • Study-hours + attendance inputs
  • Prediction button with pass probability
  • Result message
  • Explanation of how the prediction works
  • Bonus: chart of predicted class vs study hours (at the chosen attendance)
"""

import os

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -------- Page config --------
st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass / Fail Predictor")

# -------- Load model --------
# Resolve relative to this file so the app works regardless of cwd.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_model.joblib")
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
N_FEATURES = getattr(model, "n_features_in_", 2)
if N_FEATURES != 2:
    st.warning(
        "Loaded model expects "
        f"{N_FEATURES} feature(s) — it looks like an old model file. "
        "Run `python train_model.py` to retrain the 2-feature model "
        "(study_hours + attendance) for the best results."
    )


def predict_one(study_hours: float, attendance: float):
    """Return (prediction, proba_pass) handling 1- or 2-feature models."""
    if N_FEATURES == 1:
        features = [[study_hours]]
    else:
        features = [[study_hours, attendance]]
    prediction = int(model.predict(features)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(features)[0][1])
        except Exception:
            proba = None
    return prediction, proba


# -------- Inputs --------
student_name = st.text_input("Student Name", placeholder="e.g. Juan Dela Cruz")
study_hours = st.slider("Study Hours per Day", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
attendance = st.slider("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

# -------- Prediction --------
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction, proba = predict_one(study_hours, attendance)
        label = "✅ Pass" if prediction == 1 else "❌ Fail"

        st.subheader("Result")
        msg = f"**{student_name}** studying **{study_hours}** hrs/day with **{attendance:.0f}%** attendance → **{label}**"
        if proba is not None:
            msg += f" (P(pass) = {proba:.2f})"
        st.write(msg)
        if proba is not None:
            st.progress(proba)

        # -------- Explanation --------
        st.subheader("How was this prediction generated?")
        st.write(
            "The prediction is made by a **Logistic Regression** model trained on "
            "**two features**: study hours per day and attendance percentage. "
            "The model learned weights for each feature plus a bias — students with "
            "a high combination of both are predicted to pass, while low values in "
            "either feature push the prediction toward fail. Roughly, around "
            "**5+ hours/day and 65%+ attendance** is the pass region."
        )

        # -------- Bonus: Chart --------
        st.subheader("Study Hours vs Predicted Class (at this attendance)")
        st.caption(f"Holding attendance fixed at {attendance:.0f}% and varying study hours.")
        hours_range = [h * 0.5 for h in range(0, 31)]  # 0 to 15 in 0.5 steps
        predictions = [predict_one(h, attendance)[0] for h in hours_range]

        df = pd.DataFrame({"Study Hours": hours_range, "Predicted Class (0=Fail, 1=Pass)": predictions})

        fig, ax = plt.subplots(figsize=(10, 3))
        colors = ["#e74c3c" if p == 0 else "#2ecc71" for p in predictions]
        ax.bar(df["Study Hours"], df["Predicted Class (0=Fail, 1=Pass)"], color=colors, width=0.4)
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Predicted Class")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Fail", "Pass"])

        # Highlight the student's input
        ax.axvline(x=study_hours, color="#3498db", linestyle="--", linewidth=2, label=f"{student_name} ({study_hours}h)")
        ax.legend()
        st.pyplot(fig)
