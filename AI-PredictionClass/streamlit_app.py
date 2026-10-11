"""Glassmorphism Student Pass/Fail Predictor."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(
    page_title="StudentAI Predctor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- Glassmorphism styling --------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #172033;
        --muted: #69758b;
        --glass: rgba(255, 255, 255, 0.62);
        --glass-border: rgba(255, 255, 255, 0.85);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 12% 8%, rgba(176, 202, 255, .82), transparent 28%),
            radial-gradient(circle at 88% 12%, rgba(220, 190, 255, .72), transparent 27%),
            radial-gradient(circle at 72% 90%, rgba(177, 235, 224, .70), transparent 30%),
            linear-gradient(135deg, #eef3ff 0%, #f8f7ff 50%, #edf8f7 100%);
        background-attachment: fixed;
    }

    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #172033;
        letter-spacing: -0.035em;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 15px 20px;
        margin-bottom: 22px;
        background: rgba(255, 255, 255, .55);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        box-shadow: 0 10px 35px rgba(65, 81, 130, .08);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #172033;
        font-weight: 700;
        font-size: 19px;
    }

    .brand-icon {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background: rgba(255,255,255,.85);
        box-shadow: 0 5px 15px rgba(70, 82, 130, .10);
        font-size: 22px;
    }

    .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border: 1px solid rgba(255,255,255,.9);
        border-radius: 999px;
        background: rgba(255,255,255,.6);
        color: #44516a;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #16a085;
        box-shadow: 0 0 0 4px rgba(22,160,133,.12);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 34px;
        margin-bottom: 22px;
        border: 1px solid rgba(255,255,255,.8);
        border-radius: 26px;
        background: linear-gradient(115deg, rgba(255,255,255,.76), rgba(255,255,255,.36));
        box-shadow: 0 18px 50px rgba(69, 82, 130, .10);
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
    }

    .hero:after {
        content: "✦";
        position: absolute;
        right: 42px;
        top: 8px;
        font-size: 145px;
        line-height: 1;
        color: rgba(130, 151, 220, .16);
        transform: rotate(12deg);
    }

    .eyebrow {
        display: inline-block;
        margin-bottom: 12px;
        padding: 7px 11px;
        border-radius: 999px;
        background: rgba(117, 139, 220, .12);
        color: #5368b1;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .12em;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0 0 10px 0;
        font-size: clamp(28px, 4vw, 43px);
        line-height: 1.12;
    }

    .hero p {
        position: relative;
        z-index: 1;
        max-width: 600px;
        margin: 0;
        color: #69758b;
        font-size: 15px;
        line-height: 1.7;
    }

    .glass-card {
        padding: 24px;
        border: 1px solid var(--glass-border);
        border-radius: 22px;
        background: var(--glass);
        box-shadow: 0 14px 38px rgba(65, 81, 130, .08);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        min-height: 100%;
    }

    .card-kicker {
        color: #7b86a0;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 30px;
        line-height: 1.2;
        color: #182238;
        font-weight: 700;
    }

    .metric-note {
        margin-top: 7px;
        color: #78839a;
        font-size: 12px;
    }

    .section-heading {
        margin: 28px 0 14px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 23px;
        font-weight: 700;
        color: #172033;
        letter-spacing: -.03em;
    }

    .result-panel {
        text-align: center;
        padding: 24px 14px;
        border-radius: 18px;
        background: rgba(255,255,255,.53);
        border: 1px solid rgba(255,255,255,.85);
    }

    .result-icon {
        display: grid;
        place-items: center;
        width: 66px;
        height: 66px;
        margin: 0 auto 14px;
        border-radius: 22px;
        background: rgba(125, 145, 220, .13);
        font-size: 30px;
    }

    .result-pass {
        color: #087f5b;
    }

    .result-fail {
        color: #c24150;
    }

    div[data-testid="stForm"] {
        padding: 0;
        border: 0;
        background: transparent;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        border: 1px solid rgba(150, 163, 194, .35);
        border-radius: 12px;
        background: rgba(255,255,255,.72);
        min-height: 44px;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus {
        border-color: #8799e6;
        box-shadow: 0 0 0 3px rgba(135,153,230,.14);
    }

    div.stButton > button,
    div.stFormSubmitButton > button,
    div[data-testid="stDownloadButton"] button {
        min-height: 45px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,.8);
        font-weight: 700;
        transition: transform .18s ease, box-shadow .18s ease;
    }

    div.stFormSubmitButton > button,
    div.stButton > button[kind="primary"] {
        color: white;
        background: linear-gradient(115deg, #667eea, #8067d8);
        box-shadow: 0 8px 20px rgba(102,126,234,.25);
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(102,126,234,.22);
        border-color: rgba(255,255,255,.95);
    }

    div[data-testid="stMetric"] {
        padding: 18px;
        border: 1px solid rgba(255,255,255,.82);
        border-radius: 18px;
        background: rgba(255,255,255,.56);
        box-shadow: 0 8px 24px rgba(65,81,130,.05);
        backdrop-filter: blur(14px);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,.9);
        border-radius: 14px;
        overflow: hidden;
    }

    hr {
        border-color: rgba(255,255,255,.8);
    }

    .footer {
        text-align: center;
        margin-top: 35px;
        color: #8791a6;
        font-size: 12px;
    }

    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 25px 22px; }
        .hero:after { right: 10px; font-size: 90px; }
        .glass-card { padding: 19px; }
        .topbar { padding: 12px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Model loading --------------------

if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. Put "
        "`student_pass_model.joblib` beside `app.py`, then run "
        "`python train_model.py` if you have not trained it yet."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
if "latest" not in st.session_state:
    st.session_state.latest = None
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# -------------------- Header and top navigation --------------------

st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="brand-icon">🎓</div>
        <div>StudentAI <span style="font-weight:400;color:#8490a7">/ Predictor</span></div>
      </div>
      <div class="status"><span class="status-dot"></span> MODEL READY</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Main navigation",
    ["Dashboard", "Prediction History", "About Model"],
    horizontal=True,
    label_visibility="collapsed",
    key="page",
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">SMARTER LEARNING INSIGHTS</div>
      <h1>Student Performance<br>Prediction</h1>
      <p>Explore a machine-learning prediction based on average daily study hours.
      Enter student details to generate a result and review your recent predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Dashboard --------------------

if page == "Dashboard":
    history = st.session_state.history
    total = len(history)
    passes = sum(item["Prediction"] == "Pass" for item in history)
    fails = total - passes
    pass_rate = (passes / total * 100) if total else 0

    metric_cols = st.columns(4)
    metrics = [
        ("TOTAL PREDICTIONS", str(total), "Predictions this session"),
        ("PREDICTED PASS", str(passes), "Classified as pass"),
        ("PREDICTED FAIL", str(fails), "Classified as fail"),
        ("PASS SHARE", f"{pass_rate:.1f}%", "Of session predictions"),
    ]

    for col, (label, value, note) in zip(metric_cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                  <div class="card-kicker">{label}</div>
                  <div class="metric-value">{value}</div>
                  <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-heading">Make a prediction</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Student details")
        st.caption("Complete the fields below to run the saved model.")

        with st.form("prediction_form", clear_on_submit=False):
            student_name = st.text_input(
                "Student name",
                placeholder="e.g. Alex Morgan",
            )
            study_hours = st.number_input(
                "Average daily study hours",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.5,
                help="Enter a value from 0 to 24 hours.",
            )
            st.caption("Input feature: study hours per day")
            submitted = st.form_submit_button(
                "✨  Predict result",
                type="primary",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            if not student_name.strip():
                st.warning("Please enter the student's name.")
            else:
                try:
                    prediction = int(model.predict([[float(study_hours)]])[0])
                    if prediction not in (0, 1):
                        st.error("Unexpected model output. Expected 0 (Fail) or 1 (Pass).")
                    else:
                        result = "Pass" if prediction == 1 else "Fail"
                        record = {
                            "Student": student_name.strip(),
                            "Study Hours": float(study_hours),
                            "Prediction": result,
                        }
                        st.session_state.history.insert(0, record)
                        st.session_state.latest = record
                        st.rerun()
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    with right:
        latest = st.session_state.latest
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Prediction result")
        st.caption("YOUR MOST RECENT MODEL OUTPUT")

        if latest:
            if latest["Prediction"] == "Pass":
                icon = "✓"
                result_class = "result-pass"
                result_text = "Predicted Pass"
                description = "The model classified this input as Pass."
            else:
                icon = "×"
                result_class = "result-fail"
                result_text = "Predicted Fail"
                description = "The model classified this input as Fail."

            st.markdown(
                f"""
                <div class="result-panel">
                  <div class="result-icon">{icon}</div>
                  <div class="{result_class}" style="font-size:25px;font-weight:700">{result_text}</div>
                  <p style="margin:8px 0 0">{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            detail1, detail2 = st.columns(2)
            detail1.metric("Student", latest["Student"])
            detail2.metric("Study hours/day", f'{latest["Study Hours"]:.1f} h')
            st.progress(
                min(latest["Study Hours"] / 24, 1.0),
                text="Study hours compared with 24 hours",
            )
            st.caption("The progress bar shows time entered, not a pass probability.")
        else:
            st.markdown(
                """
                <div class="result-panel">
                  <div class="result-icon">✦</div>
                  <div style="font-size:22px;font-weight:700;color:#52617c">Your result appears here</div>
                  <p style="margin:8px 0 0">Submit student details to see the model's prediction.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.info("No prediction has been made in this session yet.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Study-hours analysis</div>', unsafe_allow_html=True)
    chart_left, chart_right = st.columns([1.15, 0.85], gap="large")

    hours_range = [x / 2 for x in range(49)]
    chart_df = pd.DataFrame({"Study Hours": hours_range})
    chart_df["Predicted Class"] = [
        int(model.predict([[hours]])[0]) for hours in hours_range
    ]
    chart_df["Predicted Result"] = chart_df["Predicted Class"].map({0: "Fail", 1: "Pass"})

    with chart_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Model output by study hours")
        st.caption("Class 0 = Fail · Class 1 = Pass")
        st.line_chart(
            chart_df.set_index("Study Hours")[["Predicted Class"]],
            height=290,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Prediction samples")
        st.caption("Example outputs across study-hour values")
        st.dataframe(
            chart_df[["Study Hours", "Predicted Result"]],
            use_container_width=True,
            hide_index=True,
            height=290,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Educational demonstration only. The prediction depends on the saved model "
        "and study hours alone; it is not a guarantee of actual academic performance."
    )

# -------------------- Prediction history --------------------

elif page == "Prediction History":
    st.markdown('<div class="section-heading">Prediction history</div>', unsafe_allow_html=True)
    st.write("Review predictions created during this Streamlit session.")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download history as CSV",
            data=csv,
            file_name="student_prediction_history.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.latest = None
            st.rerun()
    else:
        st.markdown(
            """
            <div class="glass-card">
              <div class="result-icon">📋</div>
              <h3>No predictions yet</h3>
              <p>Go to Dashboard and make a prediction. Your results will be listed here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------- About model --------------------

elif page == "About Model":
    st.markdown('<div class="section-heading">About the model</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
          <h3>How StudentAI works</h3>
          <p>This application loads a model that was trained earlier and saved as a Joblib file.</p>
          <ol>
            <li><b>Train:</b> <code>train_model.py</code> trains the machine-learning model.</li>
            <li><b>Save:</b> <code>joblib.dump()</code> stores the trained model.</li>
            <li><b>Load:</b> <code>joblib.load()</code> loads the saved model into this app.</li>
            <li><b>Predict:</b> The app sends the study-hours value to <code>model.predict()</code>.</li>
            <li><b>Display:</b> The predicted class is shown as Pass or Fail.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    info1, info2, info3 = st.columns(3)
    with info1:
        st.markdown(
            '<div class="glass-card"><div class="card-kicker">MODEL FILE</div>'
            '<b>student_pass_model.joblib</b></div>',
            unsafe_allow_html=True,
        )
    with info2:
        st.markdown(
            '<div class="glass-card"><div class="card-kicker">INPUT FEATURE</div>'
            '<b>Average daily study hours</b></div>',
            unsafe_allow_html=True,
        )
    with info3:
        st.markdown(
            '<div class="glass-card"><div class="card-kicker">OUTPUT CLASSES</div>'
            '<b>0 = Fail · 1 = Pass</b></div>',
            unsafe_allow_html=True,
        )

    st.warning(
        "This is an educational project, not a validated academic assessment tool. "
        "The model's quality depends on its training data. Its output is not a probability "
        "or a guaranteed result."
    )

st.markdown(
    '<div class="footer">STUDENTAI · MACHINE LEARNING DEMONSTRATION · BUILT WITH STREAMLIT</div>',
    unsafe_allow_html=True,
)
