"""Student pass/fail studio"""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).with_name("student_pass_model.joblib")
clf = joblib.load(MODEL_FILE)

st.set_page_config(page_title="Pass/Fail", layout="wide")

st.title("Pass / Fail Studio")
st.write(
    "Score a student with the saved logistic model, "
    "then see the outcome as a doughnut chart."
)
st.divider()

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Student inputs")
    full_name = st.text_input("Student name", placeholder="e.g Honeylet")
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

        # Doughnut 1: current P(pass) vs P(fail).
        current = pd.DataFrame(
            {"slice": ["Pass chance", "Fail chance"], "share": [chance, 1 - chance]}
        )
        donut_now = (
            alt.Chart(current)
            .mark_arc(innerRadius=65)
            .encode(theta="share", color="slice", tooltip=["slice", "share"])
            .properties(title="Current prediction split", height=260)
        )
        st.altair_chart(donut_now, use_container_width=True)

        with st.expander("How was this generated?", expanded=True):
            st.write(
                "Hours and attendance are standardised with the scaler stored "
                "in student_pass_model.joblib, scored by logistic regression "
                "for P(pass), then cut at 0.50 for PASS/FAIL. "
                "The file is loaded once -- no retraining here."
            )
    else:
        st.info("No prediction yet.")

st.divider()
st.subheader("Study hours vs predicted class")
st.write(
    f"Attendance fixed at {attendance}%. "
    "Every half-hour from 0 to 10 is swept through the loaded model; "
    "bars compare study hours (x) with P(pass) and predicted class (y)."
)
hour_grid = [round(x * 0.5, 1) for x in range(0, 21)]
probs = [float(clf.predict_proba([[h, attendance]])[0][1]) for h in hour_grid]
classes = [int(p >= 0.5) for p in probs]

chart_df = pd.DataFrame(
    {"study_hours": hour_grid, "P(pass)": probs, "predicted_class": classes}
)
st.bar_chart(chart_df, x="study_hours", y=["P(pass)", "predicted_class"])
st.caption(
    f"Sweep detail: {int(sum(classes))} of {len(classes)} hour steps predict PASS "
    "at 0.50 threshold."
)
