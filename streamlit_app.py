from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_pass_model.joblib"

INK = "#111111"
PAPER = "#FBF3E4"
YELLOW = "#FFE600"
PINK = "#FF5AA0"
CYAN = "#4DE1E1"
LIME = "#B6FF3B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@700&display=swap');

:root {{
    --ink: {INK};
    --paper: {PAPER};
    --yellow: {YELLOW};
    --pink: {PINK};
    --cyan: {CYAN};
    --lime: {LIME};
}}

/* Page canvas */
.stApp {{
    background:
        repeating-linear-gradient(45deg, rgba(17,17,17,0.035) 0 2px, transparent 2px 22px),
        var(--paper);
    font-family: 'Space Grotesk', system-ui, sans-serif;
}}

.block-container {{
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 3.5rem;
}}

h1, h2, h3, h4 {{
    color: var(--ink);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}}

/* Hero banner */
.hero {{
    background: var(--yellow);
    border: 4px solid var(--ink);
    border-radius: 18px;
    box-shadow: 10px 10px 0 var(--ink);
    padding: 34px 36px;
    margin-bottom: 16px;
}}
.hero h1 {{
    margin: 0;
    font-size: clamp(2rem, 4.4vw, 3.3rem);
    text-transform: uppercase;
    line-height: 0.98;
    letter-spacing: -0.03em;
}}
.hero p {{
    margin: 14px 0 0;
    font-size: 1.05rem;
    max-width: 640px;
    color: var(--ink);
}}
.badges {{ margin-top: 20px; }}
.badge {{
    display: inline-block;
    background: var(--ink);
    color: var(--paper);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 7px 12px;
    border-radius: 999px;
    margin: 0 10px 8px 0;
}}

/* Section headers */
.section-tag {{
    display: inline-flex;
    align-items: center;
    gap: 12px;
    margin: 26px 0 14px;
}}
.section-tag .num {{
    background: var(--ink);
    color: var(--lime);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 5px 12px;
    border-radius: 8px;
    border: 3px solid var(--ink);
}}
.section-tag .label {{
    font-size: 1.35rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    color: var(--ink);
}}

/* Inputs */
div[data-baseweb="input"],
div[data-baseweb="select"] {{
    border-radius: 12px !important;
}}
.stTextInput input,
.stNumberInput input {{
    background: #ffffff !important;
    border: 3px solid var(--ink) !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
}}
label, .stTextInput label p, .stNumberInput label p {{
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.78rem !important;
    color: var(--ink) !important;
}}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {{
    background: var(--pink);
    color: var(--ink);
    border: 3px solid var(--ink);
    border-radius: 12px;
    min-height: 52px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    box-shadow: 6px 6px 0 var(--ink);
    transition: transform 0.08s ease, box-shadow 0.08s ease, background 0.15s ease;
}}
.stButton > button:hover,
.stFormSubmitButton > button:hover {{
    background: var(--cyan);
    color: var(--ink);
    border: 3px solid var(--ink);
    transform: translate(3px, 3px);
    box-shadow: 3px 3px 0 var(--ink);
}}
.stButton > button:active,
.stFormSubmitButton > button:active {{
    transform: translate(6px, 6px);
    box-shadow: none;
}}

/* Metric tiles */
div[data-testid="stMetric"] {{
    background: #ffffff;
    border: 3px solid var(--ink);
    padding: 18px 20px;
    border-radius: 14px;
    box-shadow: 6px 6px 0 var(--ink);
}}
div[data-testid="stMetric"] label p {{
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--ink) !important;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: var(--ink);
}}

/* Form / bordered containers */
div[data-testid="stForm"],
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #ffffff;
    border: 3px solid var(--ink);
    border-radius: 16px;
    box-shadow: 8px 8px 0 var(--ink);
    padding: 24px;
}}

/* Alerts */
div[data-testid="stAlert"] {{
    border: 3px solid var(--ink);
    border-radius: 12px;
}}

/* Verdict card */
.verdict {{
    border: 4px solid var(--ink);
    border-radius: 18px;
    box-shadow: 10px 10px 0 var(--ink);
    padding: 26px 30px;
    margin: 8px 0 20px;
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
}}
.verdict.pass {{ background: var(--lime); }}
.verdict.fail {{ background: var(--pink); }}
.verdict .stamp {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: clamp(2.4rem, 6vw, 4rem);
    text-transform: uppercase;
    letter-spacing: -0.02em;
    line-height: 1;
    color: var(--ink);
}}
.verdict .who {{
    font-size: 1.05rem;
    font-weight: 700;
    border-left: 4px solid var(--ink);
    padding-left: 20px;
}}

/* Footer */
.footer {{
    text-align: center;
    color: var(--ink);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    margin-top: 40px;
    border-top: 3px dashed var(--ink);
    padding-top: 18px;
}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.markdown("""
<div class="hero">
    <h1>🎓 Student Pass<br>Prediction</h1>
    <p>Estimate a student's outcome from study hours and attendance
    — powered by a logistic regression model.</p>
    <div class="badges">
        <span class="badge">Logistic Regression</span>
        <span class="badge">2 Features</span>
        <span class="badge">Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

st.markdown("""
<div class="section-tag">
    <span class="num">01</span>
    <span class="label">Student Details</span>
</div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):
    name = st.text_input("Student Name", placeholder="Enter student name")

    col1, col2 = st.columns(2)

    with col1:
        hours = st.number_input(
            "Study Hours Per Day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

    with col2:
        attendance = st.number_input(
            "Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0
        )

    submitted = st.form_submit_button(
        "Predict Result",
        type="primary",
        use_container_width=True
    )

if submitted:
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        features = pd.DataFrame(
            [[hours, attendance]],
            columns=["study_hours", "attendance_percentage"]
        )

        prediction = int(model.predict(features)[0])
        label = "PASS" if prediction == 1 else "FAIL"

        verdict_class = "pass" if prediction == 1 else "fail"
        icon = "✅" if prediction == 1 else "⚠️"

        st.markdown(f"""
        <div class="verdict {verdict_class}">
            <div class="stamp">{icon} {label}</div>
            <div class="who">{name.strip()}<br>Predicted outcome</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        col1.metric("📚 Study Hours", f"{hours:.1f} hrs")
        col2.metric("📅 Attendance", f"{attendance:.0f}%")
        col3.metric("🎯 Predicted Result", label)

        st.info(
            "The model used study hours and attendance percentage "
            "to generate its prediction from patterns learned "
            "during training. This is an estimate, not a guarantee."
        )

st.markdown("""
<div class="section-tag">
    <span class="num">02</span>
    <span class="label">Impact Analysis</span>
</div>
""", unsafe_allow_html=True)

chart_hours = list(range(0, 11))
fixed_attendance = 85.0

chart_data = pd.DataFrame({
    "study_hours": chart_hours,
    "attendance_percentage": [fixed_attendance] * len(chart_hours)
})

chart_predictions = [int(value) for value in model.predict(chart_data)]
value_words = ["FAIL", "PASS"]

fig = go.Figure(
    data=[go.Scatter(
        x=chart_hours,
        y=chart_predictions,
        mode="lines+markers",
        line=dict(color=INK, width=4),
        marker=dict(size=12, color=PINK, line=dict(color=INK, width=2)),
        name="Predicted class",
        customdata=[value_words[value] for value in chart_predictions],
        hovertemplate=(
            "Study hours: %{x}<br>Outcome: %{customdata}<extra></extra>"
        )
    )]
)

fig.update_layout(
    title=dict(
        text=f"Predicted Outcome by Study Hours (Attendance fixed at {fixed_attendance:.0f}%)",
        font=dict(size=15, color=INK),
        x=0.5,
        xanchor="center"
    ),
    paper_bgcolor=PAPER,
    plot_bgcolor=PAPER,
    font=dict(family="Space Grotesk, sans-serif", color=INK),
    height=520,
    margin=dict(l=70, r=30, t=70, b=60),
    xaxis=dict(
        title=dict(text="Study Hours Per Day", font=dict(size=13, color=INK)),
        range=[-0.3, 10.3],
        tickmode="linear",
        tick0=0,
        dtick=1,
        showgrid=True,
        gridcolor="rgba(17,17,17,0.12)",
        zeroline=False,
        linecolor=INK,
        linewidth=3,
        tickfont=dict(color=INK, size=12)
    ),
    yaxis=dict(
        title=dict(text="Predicted Class", font=dict(size=13, color=INK)),
        range=[-0.25, 1.25],
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["FAIL", "PASS"],
        showgrid=True,
        gridcolor="rgba(17,17,17,0.12)",
        zeroline=False,
        linecolor=INK,
        linewidth=3,
        tickfont=dict(color=INK, size=12)
    ),
    legend=dict(
        bgcolor=YELLOW,
        bordercolor=INK,
        borderwidth=3,
        font=dict(color=INK, size=12),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(
    fig,
    width="stretch",
    config={"displayModeBar": False, "scrollZoom": True}
)

st.markdown("""
<div class="footer">
    Predictions are estimates, not guarantees of academic performance.<br>
    This application uses a small synthetic demonstration dataset.
</div>
""", unsafe_allow_html=True)
