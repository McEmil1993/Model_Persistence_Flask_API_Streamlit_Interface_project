import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        /* App background */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        }

        /* Hide default Streamlit chrome */
        #MainMenu, footer, header {visibility: hidden;}

        /* Card container */
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
            max-width: 780px;
        }

        /* Hero title */
        .hero-title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
            letter-spacing: -0.5px;
        }

        .hero-subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        /* Inputs */
        .stTextInput input, .stNumberInput input {
            background-color: rgba(30, 41, 59, 0.85) !important;
            color: #e2e8f0 !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            padding: 0.6rem 0.9rem !important;
            font-size: 1rem !important;
        }
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25) !important;
        }
        label, .stMarkdown p { color: #cbd5e1 !important; }

        /* Button */
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #38bdf8, #a78bfa);
            color: #0f172a;
            font-weight: 700;
            font-size: 1.05rem;
            padding: 0.75rem 1rem;
            border: none;
            border-radius: 12px;
            transition: transform 0.15s ease, box-shadow 0.2s ease;
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.35);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(167, 139, 250, 0.5);
        }

        /* Result cards */
        .result-card {
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            margin-top: 1.5rem;
            border: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(8px);
        }
        .result-pass {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(5, 150, 105, 0.08));
            border-left: 5px solid #10b981;
        }
        .result-fail {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.18), rgba(185, 28, 28, 0.08));
            border-left: 5px solid #ef4444;
        }
        .result-title { font-size: 1.4rem; font-weight: 700; margin: 0 0 0.4rem 0; }
        .result-pass .result-title { color: #6ee7b7; }
        .result-fail .result-title { color: #fca5a5; }
        .result-text { color: #e2e8f0; font-size: 1rem; margin: 0; }

        /* Info / explanation panel */
        .info-panel {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.1rem 1.4rem;
            margin-top: 1.25rem;
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .info-panel b { color: #7dd3fc; }

        /* Section header */
        .section-header {
            color: #e2e8f0;
            font-weight: 700;
            font-size: 1.15rem;
            margin-top: 2rem;
            margin-bottom: 0.5rem;
        }

        /* Caption */
        .stCaption, [data-testid="stCaptionContainer"] { color: #94a3b8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">🎓 Student Pass Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Enter study hours and get an instant pass prediction.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
name = st.text_input("Student name", placeholder="e.g. Alex Johnson")
hours = st.number_input(
    "Study hours per day",
    min_value=0.0,
    max_value=24.0,
    value=3.0,
    step=0.5,
    help="Range: 0 to 24 hours",
)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button("✨ Predict Result"):
    pred = int(model.predict([[hours]])[0])
    prob = float(model.predict_proba([[hours]])[0][1])
    who = name.strip() or "This student"

    if pred == 1:
        st.markdown(
            f"""
            <div class="result-card result-pass">
                <p class="result-title">✅ {who} is predicted to PASS</p>
                <p class="result-text">Probability of passing: <b>{prob:.0%}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-fail">
                <p class="result-title">⚠️ {who} is predicted to FAIL</p>
                <p class="result-text">Probability of passing: <b>{prob:.0%}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-panel">
            <b>How this was generated:</b> a logistic regression model learned from past
            examples of study hours and results. It converts your study hours into a pass
            probability and predicts <b>PASS</b> when that probability is above 50%.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Sweep chart across 0–24 hours
    # -----------------------------------------------------------------------
    sweep = pd.DataFrame({"study_hours": [h / 2 for h in range(0, 49)]})  # 0 to 24 in 0.5 steps
    sweep["predicted_class"] = model.predict(sweep[["study_hours"]].values)
    sweep["pass_probability"] = model.predict_proba(sweep[["study_hours"]].values)[:, 1]

    st.markdown('<div class="section-header">📈 Predicted Class vs Study Hours</div>', unsafe_allow_html=True)
    st.line_chart(sweep.set_index("study_hours")[["predicted_class"]], height=260)

    st.markdown('<div class="section-header">🎯 Pass Probability vs Study Hours</div>', unsafe_allow_html=True)
    st.line_chart(sweep.set_index("study_hours")[["pass_probability"]], height=260)

    st.caption(f"Your input: **{hours} hours** → predicted class **{pred}** ({prob:.0%} pass probability).")