"""Streamlit interface for the student pass-prediction model."""

import streamlit as st

from train_model import load_model, pass_probability, prediction_label


st.set_page_config(
    page_title="Studywise | Pass Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

        :root {
            --ink: #172b27;
            --muted: #71817c;
            --green: #176b52;
            --mint: #e9f4ef;
            --line: #e6ece8;
        }
        .stApp { background: #f6f8f5; color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stAppViewContainer"] > .main { background: transparent; }
        .block-container { max-width: 1050px; padding-top: 1.6rem; padding-bottom: 2rem; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); }
        .eyebrow { color: var(--green); text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 700; margin-bottom: .6rem; }
        .hero-title { font-family: 'Manrope', sans-serif; font-size: clamp(2.2rem, 4vw, 3.35rem); line-height: 1.08; letter-spacing: -.055em; font-weight: 800; color: var(--ink); margin: 0; }
        .panel-kicker { color: var(--green); font-size: .76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; margin: .8rem 0; }
        .result-empty { display: flex; height: 220px; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: var(--muted); }
        .result-label { color: var(--muted); text-transform: uppercase; letter-spacing: .1em; font-size: .72rem; font-weight: 700; }
        .result-detail { color: var(--muted); font-size: .92rem; }
        .tiny-note { color: #82908b; font-size: .78rem; line-height: 1.5; margin-top: 1.15rem; }
        div.stButton > button { background: var(--green); color: #fff; border: 0; border-radius: 11px; min-height: 3rem; font-weight: 700; transition: background .15s ease, transform .15s ease; }
        div.stButton > button:hover { background: #105a43; color: #fff; border: 0; transform: translateY(-1px); }
        div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input { border-radius: 10px; }
        div[data-testid="stProgress"] > div > div { background: var(--green); }
        div[data-testid="stProgress"] { margin-top: .5rem; }
        [data-testid="stMetric"] { background: #f4f8f5; border-radius: 14px; padding: .95rem 1rem; }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Manrope', sans-serif; }
        @media (max-width: 700px) {
            .block-container { padding-top: 1.5rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="eyebrow">STUDYWISE</div>
    <h1 class="hero-title">Student Pass Predictor</h1>
    """,
    unsafe_allow_html=True,
)
st.write("")

model = load_model()
input_col, result_col = st.columns([0.92, 1.08], gap="large")

with input_col:
    st.markdown('<div class="panel-kicker">Student details</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        name = st.text_input("Student name", placeholder="e.g. Alex Cruz")
        study_hours = st.number_input(
            "Study hours per week",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5,
        )
        attendance_percentage = st.number_input(
            "Attendance percentage",
            min_value=0,
            max_value=100,
            value=85,
            step=1,
            help="Enter a value from 0 to 100.",
        )
        submitted = st.form_submit_button("Predict result", type="primary", use_container_width=True)

with result_col:
    st.markdown('<div class="panel-kicker">Prediction</div>', unsafe_allow_html=True)

    if submitted:
        student_name = name.strip() or "Your student"
        result = prediction_label(model, study_hours)
        probability = pass_probability(model, study_hours) * 100
        if result == "Pass":
            st.success(f"{student_name} is predicted to **Pass**")
        else:
            st.warning(f"{student_name} is predicted to **Fail**")
        metric_col, hours_col, attendance_col = st.columns(3)
        metric_col.metric("Estimated pass chance", f"{probability:.1f}%")
        hours_col.metric("Study time", f"{study_hours:g} hrs / wk")
        attendance_col.metric("Attendance", f"{attendance_percentage}%")
        st.progress(int(round(probability)), text="Predicted pass probability")
        chart_width = 640
        chart_height = 220
        left, right, top, bottom = 48, 12, 12, 34
        plot_width = chart_width - left - right
        plot_height = chart_height - top - bottom
        points = " ".join(
            f"{left + hours / 100 * plot_width:.1f},"
            f"{top + (1 - pass_probability(model, hours)) * plot_height:.1f}"
            for hours in range(101)
        )
        current_x = left + study_hours / 100 * plot_width
        current_y = top + (1 - pass_probability(model, study_hours)) * plot_height
        current_x = min(left + plot_width, max(left, current_x))
        svg_chart = f"""<svg viewBox="0 0 {chart_width} {chart_height}" role="img"
            aria-label="Estimated pass chance by study hours. At {study_hours:g} hours, the estimate is {probability:.1f} percent."
            style="width:100%;height:auto;display:block">
            <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#dce5df" />
            <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#dce5df" />
            <text x="{left - 8}" y="{top + 5}" text-anchor="end" fill="#71817c" font-size="12">100%</text>
            <text x="{left - 8}" y="{top + plot_height}" text-anchor="end" fill="#71817c" font-size="12">0%</text>
            <text x="{left}" y="{chart_height - 8}" fill="#71817c" font-size="12">0 hrs</text>
            <text x="{left + plot_width}" y="{chart_height - 8}" text-anchor="end" fill="#71817c" font-size="12">100 hrs / wk</text>
            <polyline points="{points}" fill="none" stroke="#176b52" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
            <line x1="{current_x:.1f}" y1="{top}" x2="{current_x:.1f}" y2="{top + plot_height}" stroke="#e69a38" stroke-dasharray="4 4" />
            <circle cx="{current_x:.1f}" cy="{current_y:.1f}" r="5" fill="#e69a38" stroke="white" stroke-width="2" />
            </svg>"""
        st.markdown("**Pass chance by study time**")
        st.markdown(svg_chart, unsafe_allow_html=True)
        st.caption(
            f"The curve shows the model estimate across 0–100 hours. "
            f"Your input ({study_hours:g} hrs / wk) corresponds to {probability:.1f}%."
        )
        st.markdown(
            '<div class="tiny-note">This example model was trained using study-hours data. Attendance is recorded here, but does not currently affect the estimate.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-empty"><div class="result-label">Ready when you are</div><div class="result-detail">Your prediction will appear here.</div></div>',
            unsafe_allow_html=True,
        )

