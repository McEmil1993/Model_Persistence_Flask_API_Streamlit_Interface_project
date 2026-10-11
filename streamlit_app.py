"""Exercise 3 - Streamlit interface.

Run:  streamlit run streamlit_app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import model_lib  # noqa: F401

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

st.title("🎓 Student Pass Predictor")

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=4.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        X = np.array([[hours]])
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0])

        if pred == 1:
            st.success(f"{name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"{name} is predicted to FAIL (pass probability {prob:.0%}).")

        st.subheader("How was this prediction generated?")
        st.write(
            "A logistic regression model was trained on past student records "
            "(study hours and whether they passed). It turns the study hours you "
            "entered into a pass probability. If the probability is 50% or higher, "
            "the predicted class is PASS; otherwise FAIL. The model was saved with "
            "joblib, so it is loaded here and not retrained."
        )

        # Bonus: study hours vs predicted class
        st.subheader("Study hours vs predicted class")
        xs = np.array([[h / 2] for h in range(0, 25)])
        grid = pd.DataFrame({"study_hours": xs[:, 0]})
        grid["pass_probability"] = model.predict_proba(xs)
        grid["predicted_class (1=pass)"] = model.predict(xs)
        st.line_chart(grid.set_index("study_hours")[["pass_probability", "predicted_class (1=pass)"]])
        st.caption(f"{name or 'This student'} studies {hours} hours. Look where that sits on the curve.")
