"""Exercise 3 - Streamlit version (only if Streamlit installs on your device).
Otherwise use web_app.py, which does the same thing with Flask."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from model_utils import LogisticModel  # needed so joblib can load the model

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")
model = joblib.load("student_pass_model.joblib")

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", 0.0, 24.0, 3.0, 0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])
        if pred == 1:
            st.success(f"{name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"{name} is predicted to FAIL (pass probability {prob:.0%}).")
        st.subheader("How was this generated?")
        st.write(f"The app loaded a saved logistic regression model that was trained earlier. "
                 f"It used **{hours} hours** to compute a pass probability of **{prob:.0%}** and "
                 f"predicts Pass when that is 50% or more. The model was not retrained.")
        st.subheader("Study hours vs predicted class")
        xs = np.arange(0, 12.1, 0.5)
        chart = pd.DataFrame({"Pass probability": model.predict_proba(xs.reshape(-1, 1))[:, 1]}, index=xs)
        chart.index.name = "Study hours"
        st.line_chart(chart)
