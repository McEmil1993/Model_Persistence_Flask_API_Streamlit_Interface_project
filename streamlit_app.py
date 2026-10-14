"""Student pass/fail studio -- restyled Streamlit front end."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).with_name("student_pass_model.joblib")
clf = joblib.load(MODEL_FILE)

st.set_page_config(page_title="Pass/Fail Studio", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7f2;
        background-image: linear-gradient(rgba(37, 72, 62, .035) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(37, 72, 62, .035) 1px, transparent 1px);
        background-size: 28px 28px;
        color: #1d302b;
    }
    .block-container { max-width: 1180px; padding-top: 2.8rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #1d302b; letter-spacing: 0; }
    .studio-kicker {
        color: #39735e;
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }
    .studio-intro {
        color: #5b6b64;
        font-size: 1.02rem;
        max-width: 670px;
        margin: -.55rem 0 1.7rem;
    }
    [data-testid="stMetric"] {
        background: #f0f6ee;
        border-left: 4px solid #4c9b72;
        padding: 1rem 1.1rem;
        border-radius: 4px;
    }
    [data-testid="stMetricLabel"] { color: #53665e; }
    [data-testid="stMetricValue"] { color: #20583f; }
    [data-testid="stButton"] button[kind="primary"] {
        background: #20583f;
        border: 1px solid #20583f;
        color: white;
        min-height: 2.8rem;
    }
    [data-testid="stButton"] button[kind="primary"]:hover {
        background: #174631;
        border-color: #174631;
        color: white;
    }
    [data-testid="stProgress"] > div > div { background-color: #4c9b72; }
    hr { border-color: rgba(37, 72, 62, .16); }
    </style>
    <div class="studio-kicker">Student outcomes · Decision support</div>
    """,
    unsafe_allow_html=True,
)
st.title("Pass / Fail Studio")
st.markdown(
    '<p class="studio-intro">Explore how study time and attendance shape a student’s predicted outcome.</p>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("01  Student profile")
    full_name = st.text_input("Student name", placeholder="e.g. Sam Morgan")
    hours = st.slider("Study hours per week", 0.0, 12.0, 4.5, 0.5)
    attendance = st.number_input(
        "Attendance (%)", min_value=0, max_value=100, value=80, step=1
    )
    run = st.button("Run prediction", type="primary", use_container_width=True)

with right:
    st.subheader("02  Model assessment")
    if run:
        chance = float(clf.predict_proba([[hours, attendance]])[0][1])
        passed = chance >= 0.5
        verdict = "PASS" if passed else "FAIL"
        who = full_name.strip() or "This student"

        st.metric("Predicted outcome", verdict, f"{chance:.0%} chance of passing")
        st.caption("Estimated pass probability")
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
                "The file is loaded once; no retraining happens here."
            )
    else:
        st.info("Set the profile values and run an assessment to see the model’s result.")

st.divider()
st.subheader("Explore the model response")
st.write(
    f"Attendance is held at {attendance}%. Study time is swept from 0 to 12 hours per week."
)
hour_grid = [round(x * 0.5, 1) for x in range(0, 25)]
probs = [float(clf.predict_proba([[h, attendance]])[0][1]) for h in hour_grid]
classes = [int(p >= 0.5) for p in probs]
trend = pd.DataFrame(
    {"chance_pass": probs, "pass_class": classes}, index=hour_grid
)
trend.index.name = "hours"
probability_chart, outcome_chart = st.columns(2, gap="large")
with probability_chart:
    st.caption("Probability of passing")
    st.area_chart(trend[["chance_pass"]], y="chance_pass", color="#4c9b72")
with outcome_chart:
    st.caption("Predicted class · 1 = pass")
    st.bar_chart(trend[["pass_class"]], y="pass_class", color="#e38a5a")
