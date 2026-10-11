"""Exercise 3 - Streamlit interface

Run with:  streamlit run streamlit_app.py
Uses the model saved by train_model.py (student_pass_model.joblib).
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILE = "student_pass_model.joblib"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    path = os.path.join(BASE_DIR, MODEL_FILE)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


st.title("🎓 Student Pass Predictor")
st.write("Enter a student's details to predict whether they are likely to pass.")

model = load_model()
if model is None:
    st.error(f"Model file '{MODEL_FILE}' not found. Run `python train_model.py` first.")
    st.stop()

# --- Inputs ---------------------------------------------------------------
student_name = st.text_input("Student name")
study_hours = st.number_input(
    "Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5
)

# --- Prediction -----------------------------------------------------------
if st.button("Predict"):
    name = student_name.strip()
    if not name:
        st.warning("Please enter the student's name.")
    else:
        label = int(model.predict([[study_hours]])[0])
        prob_pass = float(model.predict_proba([[study_hours]])[0][1])

        if label == 1:
            st.success(f"✅ {name} is predicted to PASS (probability {prob_pass:.0%}).")
        else:
            st.error(f"❌ {name} is predicted to FAIL (pass probability {prob_pass:.0%}).")

        st.subheader("How was this prediction generated?")
        st.write(
            f"The app sent **{study_hours:g} study hours** to a logistic regression model "
            "that was trained earlier and loaded from a saved `.joblib` file. The model "
            "turns the hours into a pass probability. If that probability is 50% or "
            "higher, the predicted class is **pass**; otherwise it is **fail**. "
            "No retraining happens when you click the button."
        )

        # --- Bonus chart: study hours vs predicted class ------------------
        st.subheader("Study hours vs predicted class")
        hours_range = np.linspace(0, 12, 121)
        probs = model.predict_proba(hours_range.reshape(-1, 1))[:, 1]
        curve = pd.DataFrame({"Study hours": hours_range, "Pass probability": probs})
        st.line_chart(curve, x="Study hours", y="Pass probability")

        st.caption(
            f"{name}: {study_hours:g} hours → predicted class "
            f"{'pass (1)' if label == 1 else 'fail (0)'}. "
            "The curve shows pass probability; the class flips where it crosses 0.5."
        )
