"""Student pass/fail studio -- restyled Streamlit front end."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).with_name("student_pass_model.joblib")
clf = joblib.load(MODEL_FILE)

st.set_page_config(page_title="Pass/Fail Studio", layout="wide")

st.title("Pass / Fail Studio")
st.write(
    "Score a student with the saved logistic model. "
    "Adjust the inputs on the left, then run the prediction."
)
st.divider()

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Student inputs")
    full_name = st.text_input("Student name", placeholder="e.g. Sam Morgan")
    hours = st.slider("Study hours per week", 0.0, 12.0, 4.5, 0.5)
    attendance = st.number_input(
        "Attendance (%)", min_value=0, max_value=100, value=80, step=1
    )
    run = st.button("Run prediction", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction")
    if run:
        chance = float(clf.predict_proba([[hours, attendance]])[0][1])
        passed = chance >= 0.5
        verdict = "PASS" if passed else "FAIL"
        who = full_name.strip() or "This student"

        st.metric("Verdict", verdict, f"P(pass) = {chance:.2f}")
        st.progress(min(max(chance, 0.0), 1.0))
        if passed:
            st.success(f"{who} is predicted to {verdict}.")
        else:
            st.error(f"{who} is predicted to {verdict}.")

        with st.expander("How was this generated?", expanded=True):
            st.write(
                "Your hours and attendance are standardised with the scaler "
                "stored in student_pass_model.joblib, fed to logistic "
                "regression for P(pass), then cut at 0.50 for PASS/FAIL. "
                "The file is loaded once -- no retraining happens here."
            )
    else:
        st.info("No prediction yet. Set the inputs and click Run prediction.")

st.divider()
st.subheader("Study hours vs predicted class")
st.write(
    f"Attendance fixed at {attendance}%. Each row sweeps one hour value "
    "through the same loaded model."
)
hour_grid = [round(x * 0.5, 1) for x in range(0, 25)]
probs = [float(clf.predict_proba([[h, attendance]])[0][1]) for h in hour_grid]
classes = [int(p >= 0.5) for p in probs]
trend = pd.DataFrame(
    {"chance_pass": probs, "pass_class": classes}, index=hour_grid
)
trend.index.name = "hours"
st.area_chart(trend[["chance_pass"]])
st.bar_chart(trend[["pass_class"]])
