"""Exercise 3 - Streamlit Interface (Rayla)

A simple interface so a non-technical user can use the model.
Uses a form and session_state so the result survives Streamlit reruns.

Run:  streamlit run streamlit_app.py
"""

from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# Use a local model file if present, otherwise the one built in exercise1
MODEL_FILE = Path(__file__).resolve().parent / "rayla_pass_model.joblib"
if not MODEL_FILE.exists():
    MODEL_FILE = (
        Path(__file__).resolve().parent.parent / "exercise1" / "rayla_pass_model.joblib"
    )

model = joblib.load(MODEL_FILE)

st.set_page_config(page_title="Exam Result Predictor", page_icon=":graduation-cap:")
st.title("Exam Result Predictor")

with st.form("predict_form"):
    name = st.text_input("Student name")
    hours = st.slider("Study hours", 0.0, 24.0, 5.0, 0.5)
    submitted = st.form_submit_button("Predict")

if submitted:
    if name.strip() == "":
        st.error("Please enter the student's name.")
    else:
        features = np.array([[hours]], dtype=float)
        result = "Pass" if int(model.predict(features)[0]) == 1 else "Fail"

        st.session_state["message"] = f"{name} is predicted to {result}."
        st.session_state["explanation"] = (
            "How this works: the app loads a saved model file and applies it "
            "to the study hours you entered. Nothing is trained again."
        )

if "message" in st.session_state:
    st.subheader("Result")
    st.success(st.session_state["message"])
    st.caption(st.session_state["explanation"])
    # Bonus: chart comparing the entered study hours with the predicted class
    st.bar_chart({"Your study hours": hours, "Example Pass level": 5.0})
