"""
Exercise 3 - Streamlit interface

Run with:  streamlit run streamlit_app.py
Requires student_pass_model.joblib (created by train_model.py).
"""

from html import escape
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #172b4d;
        --muted: #65758b;
        --blue: #315ee8;
        --line: #e3eaf3;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(49, 94, 232, .08), transparent 32rem),
            #f4f7fb;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); }
    .hero {
        padding: 2rem 2.25rem;
        margin: 0 0 1.6rem;
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 20px;
        color: white;
        background: linear-gradient(118deg, #142b4d 0%, #1b3a68 58%, #315ee8 130%);
        box-shadow: 0 16px 38px rgba(23, 43, 77, .13);
    }
    .hero-kicker {
        display: inline-block;
        margin-bottom: .8rem;
        padding: .35rem .7rem;
        border: 1px solid rgba(255,255,255,.23);
        border-radius: 999px;
        color: #dce7ff;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0;
        color: #fff;
        font-size: clamp(2rem, 4vw, 2.8rem);
        letter-spacing: -.045em;
    }
    .hero p {
        max-width: 650px;
        margin: .6rem 0 0;
        color: #d5e0f2;
        font-size: 1rem;
        line-height: 1.65;
    }
    .section-label {
        margin: .15rem 0 .75rem;
        color: #73839a;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .info-card {
        height: 100%;
        padding: 1.4rem 1.5rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255,255,255,.78);
    }
    .info-card h3 {
        margin: 0 0 .55rem;
        font-size: 1.05rem;
    }
    .info-card p {
        margin: 0;
        color: var(--muted);
        font-size: .92rem;
        line-height: 1.6;
    }
    .step {
        display: flex;
        gap: .8rem;
        align-items: flex-start;
        margin-top: 1.05rem;
        color: var(--muted);
        font-size: .9rem;
        line-height: 1.5;
    }
    .step-number {
        display: grid;
        width: 1.65rem;
        height: 1.65rem;
        flex: 0 0 1.65rem;
        place-items: center;
        border-radius: 8px;
        background: #edf2ff;
        color: var(--blue);
        font-size: .75rem;
        font-weight: 700;
    }
    .result-card {
        padding: 1.5rem 1.7rem;
        border: 1px solid var(--line);
        border-left: 5px solid var(--result-color);
        border-radius: 16px;
        background: #fff;
        box-shadow: 0 8px 24px rgba(23, 43, 77, .05);
    }
    .result-label {
        color: var(--muted);
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .result-value {
        margin: .35rem 0 .2rem;
        color: var(--result-color);
        font-family: 'Manrope', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -.035em;
    }
    .result-detail { color: var(--muted); font-size: .92rem; }
    div[data-testid="stForm"] {
        padding: 1.35rem 1.5rem 1.5rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: white;
        box-shadow: 0 8px 24px rgba(23, 43, 77, .04);
    }
    div[data-testid="stForm"] label p { font-weight: 600; color: var(--ink); }
    div[data-testid="stForm"] button[kind="primary"] {
        min-height: 2.8rem;
        border: 0;
        border-radius: 9px;
        background: var(--blue);
        font-weight: 700;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover { background: #244bc6; }
    div[data-testid="stMetric"] {
        padding: 1rem 1.15rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: white;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    .footnote { color: var(--muted); font-size: .83rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="hero">
        <div class="hero-kicker">Student success analytics</div>
        <h1>Pass Predictor</h1>
        <p>Estimate a student's likelihood of passing based on their study hours.
        Clear, model-based insight to support better learning conversations.</p>
    </header>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"Model file `{MODEL_PATH.name}` not found. Run `python train_model.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    """Load the saved model once and reuse it across reruns (no retraining)."""
    return joblib.load(MODEL_PATH)


model = load_model()

input_column, overview_column = st.columns([1.12, 0.88], gap="large")

with input_column:
    st.markdown('<div class="section-label">Prediction details</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        student_name = st.text_input("Student name", placeholder="e.g. Maria Santos")
        study_hours = st.number_input(
            "Study hours per day",
            min_value=0.0,
            max_value=12.0,
            value=4.0,
            step=0.5,
            help="Enter a value from 0 to 12 hours.",
        )
        submitted = st.form_submit_button("Generate prediction", type="primary", use_container_width=True)

with overview_column:
    st.markdown(
        """
        <div class="info-card">
            <div class="section-label">How it works</div>
            <h3>A simple, transparent estimate</h3>
            <p>The predictor uses patterns learned from historical study-hour
            data. It loads the saved model and does not retrain when you submit.</p>
            <div class="step"><span class="step-number">01</span><span>Enter the student's name and typical daily study hours.</span></div>
            <div class="step"><span class="step-number">02</span><span>Review the predicted outcome and estimated pass probability.</span></div>
            <div class="step"><span class="step-number">03</span><span>Use the probability curve to see how study time relates to the estimate.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if submitted:
    if not student_name.strip():
        st.warning("Please enter the student's name first.")
        st.stop()

    predicted_class = int(model.predict([[study_hours]])[0])
    pass_probability = float(model.predict_proba([[study_hours]])[0][1])
    name = student_name.strip()
    escaped_name = escape(name)
    result_color = "#16845b" if predicted_class == 1 else "#c2414b"
    result_text = "PASS" if predicted_class == 1 else "FAIL"

    st.markdown("---")
    st.markdown('<div class="section-label">Prediction summary</div>', unsafe_allow_html=True)
    result_column, probability_column = st.columns([1.35, 0.65], gap="large")
    with result_column:
        st.markdown(
            f"""
            <div class="result-card" style="--result-color: {result_color}">
                <div class="result-label">Predicted outcome for {escaped_name}</div>
                <div class="result-value">{result_text}</div>
                <div class="result-detail">Based on {study_hours:g} hours of study per day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with probability_column:
        st.metric("Estimated chance of passing", f"{pass_probability:.0%}")

    st.markdown("")
    explanation_column, chart_column = st.columns([0.85, 1.15], gap="large")
    with explanation_column:
        st.markdown('<div class="section-label">Model insight</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if hasattr(model, "coef_"):
                coef = float(model.coef_[0][0])
                intercept = float(model.intercept_[0])
                st.write(
                    "This logistic regression model was loaded from a saved file, not retrained. "
                    f"It converts study hours into a pass probability using "
                    f"`1 / (1 + e^-({coef:.2f} × hours {intercept:+.2f}))`. "
                    f"At {study_hours:g} hours, the estimate is **{pass_probability:.0%}**. "
                    "A probability of 50% or more is classified as a pass."
                )
            else:
                st.write(
                    "The model was loaded from a saved file (not retrained) and compares "
                    "study hours with patterns learned from past students."
                )

    with chart_column:
        st.markdown('<div class="section-label">Study hours and pass probability</div>', unsafe_allow_html=True)
        grid = pd.DataFrame({"study_hours": [h / 10 for h in range(0, 121)]})
        grid["pass_probability"] = model.predict_proba(grid[["study_hours"]].values)[:, 1]
        student_point = pd.DataFrame(
            {"study_hours": [study_hours], "pass_probability": [pass_probability], "student": [name]}
        )

        curve = alt.Chart(grid).mark_line(color="#315ee8", strokeWidth=3).encode(
            x=alt.X("study_hours:Q", title="Study hours per day"),
            y=alt.Y(
                "pass_probability:Q",
                title="Probability of passing",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
        )
        threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
            strokeDash=[5, 5], color="#8c9bb0"
        ).encode(y="y:Q")
        point = alt.Chart(student_point).mark_point(
            size=180, filled=True, color=result_color, stroke="white", strokeWidth=2
        ).encode(
            x="study_hours:Q",
            y="pass_probability:Q",
            tooltip=[
                alt.Tooltip("student:N", title="Student"),
                alt.Tooltip("study_hours:Q", title="Study hours", format=".1f"),
                alt.Tooltip("pass_probability:Q", title="Pass probability", format=".0%"),
            ],
        )

        st.altair_chart((curve + threshold + point).properties(height=280), use_container_width=True)
        st.markdown(
            '<div class="footnote">Dashed line = 50% decision threshold. The marker shows this student.</div>',
            unsafe_allow_html=True,
        )
