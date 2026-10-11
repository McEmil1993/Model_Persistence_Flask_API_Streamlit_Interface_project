
"""Student Pass Predictor — interactive Streamlit dashboard."""

from pathlib import Path
from html import escape

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Outcomes | AI Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# CUSTOM UI
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --green: #24553d;
        --green-dark: #173b2a;
        --green-light: #eaf4ec;
        --bg: #f5f7f4;
        --ink: #1e3025;
        --muted: #78857b;
        --line: #e3e9e2;
    }

    .stApp {
        background: var(--bg);
        color: var(--ink);
        font-family: "Aptos", "Segoe UI", sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--ink) !important;
        letter-spacing: -0.035em;
    }

    p, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    [data-testid="stWidgetLabel"] p {
        color: var(--ink) !important;
        font-weight: 650;
    }

    /* Header */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        gap: 12px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        background: var(--green);
        color: white;
        font-size: 24px;
    }

    .brand-name {
        font-weight: 800;
        color: var(--ink);
        font-size: 1.05rem;
    }

    .brand-caption {
        color: var(--muted);
        font-size: 0.77rem;
        margin-top: 2px;
    }

    .live-tag {
        background: #e5f1e7;
        border: 1px solid #d3e6d7;
        color: #285b3a;
        border-radius: 30px;
        padding: 8px 13px;
        font-size: 0.72rem;
        font-weight: 750;
    }

    /* Hero */
    .hero {
        background: linear-gradient(125deg, #173b2a 0%, #286747 100%);
        padding: clamp(1.5rem, 4vw, 3.1rem);
        border-radius: 24px;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 15px 35px rgba(23, 59, 42, 0.12);
    }

    .hero::after {
        content: "✳";
        position: absolute;
        right: 5%;
        top: -45px;
        font-size: 220px;
        line-height: 1;
        color: rgba(255,255,255,0.055);
        pointer-events: none;
    }

    .hero-kicker {
        color: #c9e3ce;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        font-weight: 800;
        margin-bottom: 1.2rem;
        position: relative;
        z-index: 1;
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(2rem, 4.5vw, 3.5rem);
        line-height: 1.08;
        max-width: 700px;
        margin: 0 0 1rem 0;
        position: relative;
        z-index: 1;
    }

    .hero-copy {
        color: #d7e7da;
        max-width: 600px;
        font-size: 1rem;
        line-height: 1.8;
        position: relative;
        z-index: 1;
    }

    .hero-bottom {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 1.6rem;
        position: relative;
        z-index: 1;
    }

    .hero-chip {
        border: 1px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.08);
        padding: 8px 12px;
        border-radius: 9px;
        color: white;
        font-size: 0.75rem;
    }

    /* Cards */
    .panel {
        background: white;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 5px 22px rgba(25, 55, 35, 0.025);
    }

    .eyebrow {
        color: #558165;
        font-size: 0.7rem;
        letter-spacing: 0.13em;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }

    .panel-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--ink);
        letter-spacing: -0.035em;
        margin-bottom: 0.4rem;
    }

    .panel-copy {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.7;
        margin-bottom: 1.3rem;
    }

    .stTextInput input,
    .stNumberInput input {
        background: #fbfcfa !important;
        border-radius: 10px !important;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: var(--green) !important;
        box-shadow: 0 0 0 2px rgba(36,85,61,0.1) !important;
    }

    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        min-height: 3rem;
        border-radius: 10px;
        font-weight: 750;
        transition: 0.18s ease;
    }

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        background: var(--green);
        color: white;
        border: 1px solid var(--green);
    }

    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button:hover {
        background: var(--green-dark);
        border-color: var(--green-dark);
        color: white;
        transform: translateY(-1px);
    }

    /* Result */
    .result-card {
        padding: 1.5rem;
        background: white;
        border: 1px solid var(--line);
        border-radius: 18px;
        margin-top: 1.3rem;
    }

    .result-pass {
        background: #eaf5ed;
        border-color: #cfe6d5;
    }

    .result-fail {
        background: #fff0ed;
        border-color: #f1d7d0;
    }

    .result-heading {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        font-weight: 800;
        color: var(--muted);
        margin-bottom: 0.7rem;
    }

    .result-name {
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--ink);
        overflow-wrap: anywhere;
        margin-bottom: 0.65rem;
    }

    .result-status {
        font-size: 1.65rem;
        font-weight: 850;
        margin: 0.3rem 0;
    }

    .result-pass .result-status {
        color: #22663b;
    }

    .result-fail .result-status {
        color: #a34235;
    }

    .result-description {
        font-size: 0.84rem;
        line-height: 1.7;
        color: #66766a;
        margin-top: 0.6rem;
    }

    /* Section and footer */
    .section-intro {
        margin: 2.2rem 0 1rem;
    }

    .section-kicker {
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        font-weight: 800;
        color: #558165;
        margin-bottom: 0.4rem;
    }

    .section-description {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.7;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        padding: 1rem;
        border-radius: 14px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    .footer {
        border-top: 1px solid var(--line);
        margin-top: 2rem;
        padding-top: 1.2rem;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.9;
    }

    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 0.8rem 2rem;
        }

        .hero {
            border-radius: 17px;
        }

        .live-tag {
            font-size: 0.62rem;
            padding: 7px 9px;
        }

        .panel {
            padding: 1.1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. "
        "Run `python train_model.py` first."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()

# --------------------------------------------------
# HEADER AND HERO
# --------------------------------------------------
st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-icon">🎓</div>
            <div>
                <div class="brand-name">Student Outcomes</div>
                <div class="brand-caption">Learning & analytics</div>
            </div>
        </div>
        <div class="live-tag">✦ AI LEARNING LAB</div>
    </div>

    <div class="hero">
        <div class="hero-kicker">YOUR LEARNING JOURNEY · 01</div>
        <h1>Small steps today.<br>Stronger possibilities tomorrow.</h1>
        <div class="hero-copy">
            Discover how daily study hours influence a machine-learning
            model's prediction of student outcomes. Enter your details,
            explore the results, and learn something new.
        </div>
        <div class="hero-bottom">
            <div class="hero-chip">🎯 Personalized prediction</div>
            <div class="hero-chip">📊 Interactive model explorer</div>
            <div class="hero-chip">⚡ Instant results</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# QUICK OVERVIEW
# --------------------------------------------------
overview1, overview2, overview3 = st.columns(3)

with overview1:
    st.metric("Model input", "Study hours", "Per day")

with overview2:
    st.metric("Possible outcomes", "Pass / Fail", "Classification")

with overview3:
    st.metric("Prediction speed", "Instant", "On submission")

# --------------------------------------------------
# PREDICTION FORM
# --------------------------------------------------
st.markdown(
    """
    <div class="section-intro">
        <div class="section-kicker">02 / PREDICTION STUDIO</div>
        <div class="panel-title">Let's explore your outcome</div>
        <div class="section-description">
            Enter a name and choose the average number of hours
            studied per day to see the model's prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            """
            <div class="eyebrow">STUDENT PROFILE</div>
            <div class="panel-title">Student details</div>
            <div class="panel-copy">
                A few details are all you need to get started.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("prediction_form"):
            student_name = st.text_input(
                "Student name",
                placeholder="e.g., Alex",
                max_chars=100,
            )

            study_hours = st.slider(
                "Average study hours per day",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.5,
                help="Choose the average time spent studying each day.",
            )

            # Visual progress bar
            st.progress(
                int((study_hours / 24) * 100),
                text=f"{study_hours:g} of 24 hours selected",
            )

            st.caption(
                "💡 Tip: Try different study-hour values to explore "
                "how the saved model responds."
            )

            submitted = st.form_submit_button(
                "✨  Generate my prediction",
                type="primary",
                use_container_width=True,
            )

with right:
    st.markdown(
        """
        <div class="panel">
            <div class="eyebrow">HOW IT WORKS</div>
            <div class="panel-title">Your learning, visualized.</div>
            <div class="panel-copy">
                Explore a simple machine-learning workflow in three steps.
            </div>
            <div style="display:flex;gap:13px;margin:1.3rem 0;">
                <div style="background:#eaf4ec;border-radius:12px;
                            min-width:44px;height:44px;display:flex;
                            align-items:center;justify-content:center;
                            font-size:20px;">✍️</div>
                <div>
                    <div style="font-weight:750;color:#1e3025;">
                        01. Enter your details
                    </div>
                    <div style="font-size:0.83rem;color:#78857b;
                                line-height:1.7;">
                        Provide a name and daily study hours.
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:13px;margin:1.3rem 0;">
                <div style="background:#eaf4ec;border-radius:12px;
                            min-width:44px;height:44px;display:flex;
                            align-items:center;justify-content:center;
                            font-size:20px;">🧠</div>
                <div>
                    <div style="font-weight:750;color:#1e3025;">
                        02. Run the trained model
                    </div>
                    <div style="font-size:0.83rem;color:#78857b;
                                line-height:1.7;">
                        The saved model processes your study-hour value.
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:13px;margin:1.3rem 0 0;">
                <div style="background:#eaf4ec;border-radius:12px;
                            min-width:44px;height:44px;display:flex;
                            align-items:center;justify-content:center;
                            font-size:20px;">📈</div>
                <div>
                    <div style="font-weight:750;color:#1e3025;">
                        03. Explore the result
                    </div>
                    <div style="font-size:0.83rem;color:#78857b;
                                line-height:1.7;">
                        Review the predicted class and compare other inputs.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------
if submitted:
    if not student_name.strip():
        st.warning("Please enter the student's name before predicting.")
    else:
        try:
            prediction = int(
                model.predict(pd.DataFrame(
                    {"StudyHours": [float(study_hours)]}
                ))[0]
            )

            result = "Pass" if prediction == 1 else "Fail"
            safe_name = escape(student_name.strip())

            result_class = (
                "result-pass" if result == "Pass" else "result-fail"
            )

            icon = "🎉" if result == "Pass" else "📘"

            description = (
                "The model predicts a passing outcome for this input. "
                "Keep building consistent learning habits."
                if result == "Pass"
                else
                "The model predicts a failing outcome for this input. "
                "Consider reviewing study habits and seeking support."
            )

            st.markdown(
                f"""
                <div class="result-card {result_class}">
                    <div class="result-heading">
                        03 / YOUR PREDICTION RESULT
                    </div>
                    <div class="result-name">{safe_name}</div>
                    <div class="result-status">{icon} {result}</div>
                    <div class="result-description">
                        {description}<br>
                        Study time provided: {study_hours:g} hours per day.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "This is a model prediction, not a judgment of a "
                "student's ability or a guarantee of actual results."
            )

        except Exception as exc:
            st.error(
                "Prediction failed. Check whether the input features "
                f"match your trained model. Details: {exc}"
            )

# --------------------------------------------------
# MODEL EXPLORER
# --------------------------------------------------
st.markdown(
    """
    <div class="section-intro">
        <div class="section-kicker">04 / MODEL EXPLORER</div>
        <div class="panel-title">What happens when study hours change?</div>
        <div class="section-description">
            Compare predicted outcomes across a range of daily study hours.
            Every result below comes from the same saved model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hours_values = [
    0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4,
    4.5, 5, 5.5, 6, 6.5, 7, 8,
]

try:
    preview_input = pd.DataFrame({"StudyHours": hours_values})
    preview_predictions = model.predict(preview_input)

    chart_data = pd.DataFrame({
        "Study hours": hours_values,
        "Predicted outcome": [
            "Pass" if int(value) == 1 else "Fail"
            for value in preview_predictions
        ],
        "Predicted class": [
            int(value) for value in preview_predictions
        ],
    })

    table_col = st.columns(1)[0]

    with table_col:
        with st.container(border=True):
            st.markdown("**Prediction table**")
            st.caption("Review the model's class at each selected value.")

            st.dataframe(
                chart_data[["Study hours", "Predicted outcome"]],
                use_container_width=True,
                hide_index=True,
                height=380,
                column_config={
                    "Study hours": st.column_config.NumberColumn(
                        "Hours / day",
                        format="%.1f",
                    ),
                    "Predicted outcome": st.column_config.TextColumn(
                        "Outcome",
                    ),
                },
            )

except Exception as exc:
    st.error(
        "The model explorer could not generate predictions. "
        f"Check the model's expected input features. Details: {exc}"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <strong style="color:#1e3025;">🎓 About Student Outcomes</strong><br>
        An educational machine-learning demonstration using a saved
        classification model. Predictions depend on the training data
        and the features provided. This application uses study hours
        alone and does not measure a student's complete academic ability.
        Results should not be treated as definitive academic judgments.
        <br><br>
        <span>Built for learning · Student Pass/Fail Predictor</span>
    </div>
    """,
    unsafe_allow_html=True,
)
