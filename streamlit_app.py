
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# CUSTOM CSS — GOLD AND BLACK THEME
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --background: #080808;
        --panel: #141414;
        --panel-light: #1c1a14;
        --green: #d4af37;
        --green-soft: #f5df91;
        --text: #f5f1e5;
        --muted: #b8b09a;
        --border: rgba(212, 175, 55, 0.24);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, rgba(212, 175, 55, 0.12), transparent 42%),
            radial-gradient(ellipse at 95% 25%, rgba(212, 175, 55, 0.08), transparent 35%),
            var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 8, 8, 0.92);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 13px 18px;
        margin-bottom: 30px;
        background: rgba(20, 20, 20, 0.92);
        border: 1px solid var(--border);
        border-radius: 14px;
    }

    .brand {
        color: var(--green);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 1.2px;
    }

    .system-status {
        color: var(--green-soft);
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.7px;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: var(--green);
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.75);
    }

    .hero {
        padding: 10px 0 25px 0;
    }

    .eyebrow {
        color: var(--green);
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .hero h1 {
        color: var(--text);
        font-size: clamp(32px, 5vw, 51px);
        line-height: 1.13;
        letter-spacing: -1.8px;
        font-weight: 800;
        margin: 0;
    }

    .hero h1 span {
        color: var(--green);
    }

    .hero p {
        color: var(--muted);
        font-size: 15px;
        line-height: 1.8;
        max-width: 650px;
        margin-top: 15px;
    }

    .panel {
        background: linear-gradient(145deg, rgba(28, 27, 22, 0.98), rgba(12, 12, 12, 0.98));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 25px;
        min-height: 100%;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.13);
    }

    .panel-heading {
        color: var(--text);
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .panel-description {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
        margin-bottom: 22px;
    }

    .section-label {
        color: var(--green);
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.3px;
        margin: 4px 0 12px 0;
    }

    .metric-card {
        background: rgba(20, 20, 20, 0.96);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 19px;
        min-height: 110px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 13px;
    }

    .metric-value {
        color: var(--green);
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .metric-note {
        color: var(--muted);
        font-size: 11px;
        margin-top: 7px;
    }

    .result-card {
        padding: 22px;
        border-radius: 14px;
        margin: 12px 0 18px 0;
        background: rgba(212, 175, 55, 0.08);
        border: 1px solid rgba(212, 175, 55, 0.45);
    }

    .result-card.fail {
        background: rgba(255, 87, 87, 0.07);
        border-color: rgba(255, 87, 87, 0.4);
    }

    .result-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 1.5px;
        color: var(--muted);
        margin-bottom: 10px;
    }

    .result-title {
        color: var(--green);
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .result-card.fail .result-title {
        color: #ff7777;
    }

    .result-description {
        color: var(--text);
        font-size: 13px;
        line-height: 1.7;
    }

    .terminal-note {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--muted);
        background: rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(212, 175, 55, 0.15);
        padding: 13px;
        border-radius: 10px;
        line-height: 1.8;
        overflow-wrap: anywhere;
    }

    .footer {
        border-top: 1px solid var(--border);
        margin-top: 35px;
        padding-top: 20px;
        color: var(--muted);
        font-size: 11px;
        text-align: center;
        line-height: 1.8;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background: #101010 !important;
        color: var(--text) !important;
        border: 1px solid rgba(212, 175, 55, 0.30) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--green) !important;
        box-shadow: 0 0 0 1px var(--green) !important;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label {
        color: #e8dba8 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }

    div.stButton > button {
        width: 100%;
        min-height: 48px;
        border: 1px solid var(--green) !important;
        border-radius: 10px !important;
        background: var(--green) !important;
        color: #080808 !important;
        font-weight: 800 !important;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: #f5df91 !important;
        border-color: #f5df91 !important;
        box-shadow: 0 0 22px rgba(212, 175, 55, 0.22);
        transform: translateY(-1px);
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    div[data-testid="stProgress"] > div > div {
        background-color: var(--green);
    }

    [data-testid="stMetricValue"] {
        color: var(--green);
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }

        .panel {
            padding: 18px;
        }

        .topbar {
            padding: 12px;
        }

        .system-status {
            font-size: 9px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD THE SAVED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please run train_model.py first and make sure "
        "student_pass_model.joblib is in the same folder as app.py."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="topbar">
        <div class="brand">[ SP.PREDICTOR ]</div>
        <div class="system-status">
            <span class="status-dot"></span>SYSTEM ONLINE
        </div>
    </div>

    <div class="hero">
        <div class="eyebrow">// MACHINE LEARNING • STUDENT ANALYTICS</div>
        <h1>Student Pass<br><span>Predictor_</span></h1>
        <p>
            Analyze a student's study hours and generate a prediction
            using your trained machine-learning model. Enter the student's
            information below to begin.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# TOP METRIC CARDS
# --------------------------------------------------

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">MODEL STATUS</div>
            <div class="metric-value">READY</div>
            <div class="metric-note">Saved model loaded</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">INPUT VARIABLE</div>
            <div class="metric-value">STUDY HRS</div>
            <div class="metric-note">Hours studied per day</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">PREDICTION TYPE</div>
            <div class="metric-value">PASS / FAIL</div>
            <div class="metric-note">Classification output</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.write("")

# --------------------------------------------------
# MAIN INPUT AND RESULT PANELS
# --------------------------------------------------

left_column, right_column = st.columns([1, 1.15], gap="large")

with left_column:
    st.markdown(
        """
        <div class="panel">
            <div class="section-label">01 / INPUT CONFIGURATION</div>
            <div class="panel-heading">Student Information</div>
            <div class="panel-description">
                Provide the student's name and daily study hours.
                Fields marked below are required to run a prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Inputs remain outside the HTML panel so Streamlit can handle them.
    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student's full name",
        key="student_name",
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5,
        help="Enter a value from 0 to 24 hours.",
        key="study_hours",
    )

    st.markdown(
        '<div class="section-label">STUDY TIME INDICATOR</div>',
        unsafe_allow_html=True,
    )

    st.progress(
        int((study_hours / 24.0) * 100),
        text=f"{study_hours:g} of 24 hours entered",
    )

    if study_hours < 2:
        study_level = "LOW STUDY TIME"
        study_description = "A relatively small amount of daily study time."
    elif study_hours < 6:
        study_level = "MODERATE STUDY TIME"
        study_description = "A moderate amount of daily study time."
    else:
        study_level = "HIGH STUDY TIME"
        study_description = "A comparatively high amount of daily study time."

    st.caption(f"{study_level} — {study_description}")

    predict_clicked = st.button(
        "⚡  RUN PREDICTION",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="terminal-note">
            > INPUT: STUDENT PROFILE<br>
            > FEATURE: STUDY HOURS<br>
            > OUTPUT: PASS / FAIL CLASSIFICATION
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_column:
    st.markdown(
        """
        <div class="panel">
            <div class="section-label">02 / ANALYSIS OUTPUT</div>
            <div class="panel-heading">Prediction Results</div>
            <div class="panel-description">
                Your model's output and study-time summary will appear here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predict_clicked:
        if not student_name.strip():
            st.warning("Please enter the student's name before predicting.")
        else:
            try:
                # Keep the input format compatible with your trained model.
                prediction = int(model.predict([[study_hours]])[0])
                label = "PASS" if prediction == 1 else "FAIL"
                clean_name = student_name.strip()

                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">CLASSIFICATION COMPLETE</div>
                            <div class="result-title">✓ {label}</div>
                            <div class="result-description">
                                <b>{clean_name}</b> is predicted to pass
                                based on the supplied study hours and the
                                saved machine-learning model.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card fail">
                            <div class="result-label">CLASSIFICATION COMPLETE</div>
                            <div class="result-title">✕ {label}</div>
                            <div class="result-description">
                                <b>{clean_name}</b> is predicted to fail
                                based on the supplied study hours and the
                                saved machine-learning model.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.metric(
                        label="Study Hours",
                        value=f"{study_hours:g} hrs",
                    )

                with result_col2:
                    st.metric(
                        label="Predicted Class",
                        value=label,
                    )

                st.markdown(
                    '<div class="section-label">STUDY TIME BREAKDOWN</div>',
                    unsafe_allow_html=True,
                )

                # This chart shows entered study time versus the remaining
                # hours in a 24-hour day. It does not represent model confidence.
                chart_data = pd.DataFrame(
                    {
                        "Hours": [
                            study_hours,
                            24.0 - study_hours,
                        ]
                    },
                    index=[
                        "Study time",
                        "Remaining time",
                    ],
                )

                st.bar_chart(
                    chart_data,
                    color="#d4af37",
                    horizontal=True,
                )

                st.caption(
                    "The chart visualizes the entered hours within a 24-hour "
                    "day. It is not a probability or confidence score."
                )

                st.markdown(
                    f"""
                    <div class="terminal-note">
                        > STATUS: PREDICTION COMPLETE<br>
                        > STUDENT: {clean_name}<br>
                        > INPUT HOURS: {study_hours:g}<br>
                        > RESULT: {label}<br>
                        > MODEL: student_pass_model.joblib
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.info(
                    "This result was generated using the saved model. "
                    "The model was not retrained during this prediction. "
                    "The output is an estimate, not a guarantee of academic performance."
                )

            except Exception as exc:
                st.error(
                    "The model could not generate a prediction. "
                    f"Check that the saved model accepts study hours as input. Details: {exc}"
                )

    else:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">AWAITING INPUT</div>
                <div class="result-title">System Idle_</div>
                <div class="result-description">
                    Complete the student information on the left, then click
                    <b>RUN PREDICTION</b> to view the model's classification.
                </div>
            </div>

            <div class="terminal-note">
                > MODEL: LOADED<br>
                > WAITING FOR STUDENT DATA...<br>
                > READY FOR ANALYSIS
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        STUDENT PASS PREDICTOR · MACHINE LEARNING PROJECT<br>
        Educational support tool — predictions depend on the training data
        and the saved model.
    </div>
    """,
    unsafe_allow_html=True,
)