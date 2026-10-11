"""StudentIQ — Premium Student Pass/Fail Predictor."""

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(
    page_title="StudentIQ | Pass Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# Session state
# --------------------------------------------------
st.session_state.setdefault("theme", "Light")
st.session_state.setdefault("prediction_history", [])
st.session_state.setdefault("form_version", 0)


def new_prediction():
    """Start a fresh prediction without deleting history."""
    st.session_state.form_version += 1
    st.session_state.pop("latest_prediction", None)
    st.rerun()


# --------------------------------------------------
# Light / Dark theme color tokens
# --------------------------------------------------
dark = st.session_state.theme == "Dark"

if dark:
    C = {
        "bg": "#090d16",
        "surface": "#101726",
        "surface2": "#162033",
        "ink": "#f3f6ff",
        "muted": "#919fb8",
        "line": "#22304a",
        "input": "#0d1320",
        "accent": "#818cf8",
        "accent2": "#6366f1",
        "hero1": "#131b31",
        "hero2": "#281e59",
        "pass_bg": "#0e2320",
        "pass_line": "#174f42",
        "pass": "#34d399",
        "fail_bg": "#29141d",
        "fail_line": "#662435",
        "fail": "#fb7185",
    }
else:
    C = {
        "bg": "#f8fafc",
        "surface": "#ffffff",
        "surface2": "#f1f5f9",
        "ink": "#0f172a",
        "muted": "#64748b",
        "line": "#e2e8f0",
        "input": "#f8fafc",
        "accent": "#6366f1",
        "accent2": "#4f46e5",
        "hero1": "#0f172a",
        "hero2": "#312e81",
        "pass_bg": "#f0fdf4",
        "pass_line": "#bbf7d0",
        "pass": "#16a34a",
        "fail_bg": "#fff1f2",
        "fail_line": "#fecdd3",
        "fail": "#e11d48",
    }

# --------------------------------------------------
# Premium CSS design system with advanced animations
# --------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {{
        --bg: {C['bg']};
        --surface: {C['surface']};
        --surface2: {C['surface2']};
        --ink: {C['ink']};
        --muted: {C['muted']};
        --line: {C['line']};
        --input: {C['input']};
        --accent: {C['accent']};
        --accent2: {C['accent2']};
    }}

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    .stApp {{
        background: var(--bg);
        color: var(--ink);
    }}

    .block-container {{
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }}

    #MainMenu, footer, header {{
        visibility: hidden;
    }}

    [data-testid="stToolbar"] {{
        visibility: hidden;
        height: 0;
    }}

    h1, h2, h3, .display {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--ink) !important;
    }}

    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes pulseGlow {{
        0% {{ box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }}
        70% {{ box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }}
    }}

    .animate-fade {{
        animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}

    .masthead {{
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 0 0 24px;
        animation: fadeIn 0.3s ease-out;
    }}

    .brand-mark {{
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        color: #ffffff;
        font-size: 22px;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.25);
    }}

    .brand-name {{
        font: 700 22px 'Space Grotesk', sans-serif;
        letter-spacing: -.7px;
        color: var(--ink);
    }}

    .brand-caption {{
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-top: 2px;
        font-weight: 600;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        padding: 40px 42px;
        border-radius: 28px;
        margin: 4px 0 28px;
        color: white;
        background: linear-gradient(
            120deg,
            {C['hero1']} 0%,
            #1e293b 50%,
            {C['hero2']} 100%
        );
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
        animation: fadeIn 0.4s ease-out;
    }}

    .hero:after {{
        content: "";
        position: absolute;
        width: 320px;
        height: 320px;
        right: -70px;
        top: -160px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,.15);
        box-shadow: 
            0 0 0 35px rgba(255,255,255,.03),
            0 0 0 75px rgba(255,255,255,.015);
        pointer-events: none;
    }}

    .eyebrow {{
        color: #a5b4fc;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2.2px;
        text-transform: uppercase;
    }}

    .hero h1 {{
        color: #fff !important;
        font-size: clamp(32px, 3.2vw, 46px);
        line-height: 1.1;
        letter-spacing: -1.2px;
        margin: 12px 0 12px;
    }}

    .hero p {{
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.65;
        max-width: 680px;
        margin: 0;
    }}

    .hero-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-top: 22px;
        border: 1px solid rgba(255,255,255,.2);
        background: rgba(255,255,255,.08);
        backdrop-filter: blur(8px);
        color: #f8fafc;
        padding: 7px 14px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 500;
    }}

    .section-heading {{
        font: 700 22px 'Space Grotesk', sans-serif;
        color: var(--ink);
        letter-spacing: -.5px;
        margin: 4px 0;
    }}

    .section-copy {{
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 18px;
    }}

    .panel {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 26px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    .panel:hover {{
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.07);
    }}

    .panel-title {{
        font: 700 17px 'Space Grotesk', sans-serif;
        color: var(--ink);
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .panel-copy {{
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 20px;
        line-height: 1.6;
    }}

    .metric {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 20px;
        min-height: 120px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.03);
        transition: all 0.2s ease;
    }}

    .metric:hover {{
        transform: translateY(-2px);
        border-color: var(--accent);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.08);
    }}

    .metric-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
    }}

    .metric-label {{
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .8px;
    }}

    .metric-icon {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 10px;
        background: var(--surface2);
        color: var(--accent);
        font-size: 14px;
        font-weight: 700;
    }}

    .metric-value {{
        font: 700 30px 'Space Grotesk', sans-serif;
        color: var(--ink);
        margin-top: 10px;
        letter-spacing: -.8px;
    }}

    .metric-note {{
        font-size: 11px;
        color: var(--muted);
        margin-top: 4px;
    }}

    .result-card {{
        border-radius: 18px;
        padding: 22px;
        margin-top: 18px;
        border: 1px solid;
        animation: fadeIn 0.3s ease-out;
    }}

    .result-pass {{
        background: {C['pass_bg']};
        border-color: {C['pass_line']};
    }}

    .result-fail {{
        background: {C['fail_bg']};
        border-color: {C['fail_line']};
    }}

    .result-label {{
        color: var(--muted);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }}

    .result-value {{
        font: 700 32px 'Space Grotesk', sans-serif;
        margin: 8px 0;
    }}

    .result-desc {{
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
    }}

    div[data-testid="stForm"] {{
        border: 0;
        background: transparent;
        padding: 0;
    }}

    .stTextInput label,
    .stNumberInput label {{
        color: var(--ink) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}

    .stTextInput input,
    .stNumberInput input {{
        background: var(--input) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        min-height: 46px !important;
        color: var(--ink) !important;
        transition: all 0.2s ease;
    }}

    .stTextInput input:focus,
    .stNumberInput input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px rgba(99,102,241,.15) !important;
    }}

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {{
        border-radius: 12px;
        min-height: 46px;
        font-weight: 700;
        border: 1px solid var(--line);
        background: var(--surface2);
        color: var(--ink);
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        border-color: var(--accent);
        color: var(--accent);
        transform: translateY(-1px);
    }}

    .stFormSubmitButton > button {{
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        color: #fff;
        border: 0;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.25);
    }}

    .stFormSubmitButton > button:hover {{
        opacity: 0.95;
        color: #fff;
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.35);
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: var(--surface2);
        padding: 6px;
        border-radius: 14px;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 10px 18px;
        color: var(--muted);
        font-weight: 600;
    }}

    .stTabs [aria-selected="true"] {{
        background: var(--surface) !important;
        color: var(--ink) !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
    }}

    hr {{
        border-color: var(--line) !important;
    }}

    div[data-testid="stCaptionContainer"] p {{
        color: var(--muted);
        font-size: 11px;
    }}

    .footer-note {{
        color: var(--muted);
        font-size: 11px;
        line-height: 1.7;
        margin-top: 32px;
        padding-top: 20px;
        border-top: 1px solid var(--line);
    }}

    @media (max-width: 750px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        .hero {{
            padding: 28px 22px;
            border-radius: 22px;
        }}
        .panel {{
            padding: 20px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load saved model
# --------------------------------------------------
if not MODEL_PATH.exists():
    st.markdown(
        """
        <div class="hero animate-fade">
          <div class="eyebrow">STUDENTIQ · ACADEMIC ANALYTICS</div>
          <h1>Your prediction workspace</h1>
          <p>Train and save the model once, then return here to make predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("Model file not found: student_pass_model.joblib")
    st.code(
        "python train_model.py\nstreamlit run app.py",
        language="bash",
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

# --------------------------------------------------
# Header and theme switcher
# --------------------------------------------------
header_left, header_right = st.columns(
    [5, 1.4],
    vertical_alignment="center",
)

with header_left:
    st.markdown(
        """
        <div class="masthead">
          <div class="brand-mark">S</div>
          <div>
            <div class="brand-name">StudentIQ</div>
            <div class="brand-caption">
                Academic intelligence workspace
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    chosen_theme = st.selectbox(
        "Appearance",
        ["Light", "Dark"],
        index=1 if dark else 0,
        label_visibility="collapsed",
        key="appearance_selector",
    )

    if chosen_theme != st.session_state.theme:
        st.session_state.theme = chosen_theme
        st.rerun()

# --------------------------------------------------
# Hero
# --------------------------------------------------
st.markdown(
    """
    <div class="hero animate-fade">
      <div class="eyebrow">STUDENTIQ · MODEL WORKSPACE</div>
      <h1>Better insights.<br>One prediction at a time.</h1>
      <p>
        Explore how daily study habits influence your saved
        machine-learning model's pass/fail classification in
        a clean, focused workspace.
      </p>
      <span class="hero-chip">
        ✦ &nbsp; Saved model · Ready to predict
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Analytics metrics
# --------------------------------------------------
history = st.session_state.prediction_history
total = len(history)

passes = sum(
    item["prediction"] == "Pass"
    for item in history
)

fails = total - passes
pass_rate = (passes / total * 100) if total else 0

metrics = [
    ("PREDICTIONS RUN", str(total), "This session", "↗"),
    ("PREDICTED PASS", str(passes), "Model classified as pass", "✓"),
    ("PREDICTED FAIL", str(fails), "Model classified as fail", "!"),
    (
        "SESSION PASS RATE",
        f"{pass_rate:.0f}%" if total else "—",
        "Not a real class pass rate",
        "%",
    ),
]

metric_cols = st.columns(4, gap="medium")

for col, (label, value, note, symbol) in zip(
    metric_cols, metrics
):
    with col:
        st.markdown(
            f"""
            <div class="metric animate-fade">
              <div class="metric-top">
                <div class="metric-label">{label}</div>
                <div class="metric-icon">{symbol}</div>
              </div>
              <div class="metric-value">{value}</div>
              <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------
# Prediction studio heading and New Prediction button
# --------------------------------------------------
st.write("")

section_left, section_right = st.columns(
    [4, 1.2],
    vertical_alignment="center",
)

with section_left:
    st.markdown(
        """
        <div class="section-heading">Prediction studio</div>
        <div class="section-copy">
            Enter a student and evaluate a study-hours scenario.
        </div>
        """,
        unsafe_allow_html=True,
    )

with section_right:
    if st.button(
        "＋ &nbsp;New prediction",
        use_container_width=True,
        key="new_prediction_top",
    ):
        new_prediction()

# --------------------------------------------------
# Prediction form and model response
# --------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">✦ &nbsp; Create a prediction</div>
          <div class="panel-copy">
            Only average daily study hours are used by this model.
          </div>
        """,
        unsafe_allow_html=True,
    )

    form_key = st.session_state.form_version

    with st.form(
        f"prediction_form_{form_key}",
        clear_on_submit=False,
    ):
        student_name = st.text_input(
            "Student name",
            placeholder="Enter student name",
            key=f"student_name_{form_key}",
        )

        study_hours = st.number_input(
            "Average study hours per day",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5,
            help=(
                "Enter the average amount of time spent "
                "studying on a typical day."
            ),
            key=f"study_hours_{form_key}",
        )

        st.caption("Input range: 0–24 hours per day.")

        submitted = st.form_submit_button(
            "Run prediction &nbsp;→",
            use_container_width=True,
        )

    if submitted:
        if not student_name.strip():
            st.warning(
                "Enter a student name before running the prediction."
            )
        else:
            try:
                prediction = int(
                    model.predict([[float(study_hours)]])[0]
                )

                result = "Pass" if prediction == 1 else "Fail"

                item = {
                    "student": student_name.strip(),
                    "study_hours": float(study_hours),
                    "prediction": result,
                    "created_at": datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                }

                st.session_state.prediction_history.insert(
                    0, item
                )
                st.session_state.latest_prediction = item

                st.rerun()

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    latest = st.session_state.get("latest_prediction")

    if latest:
        passed = latest["prediction"] == "Pass"

        result_class = (
            "result-pass" if passed else "result-fail"
        )
        result_color = (
            C["pass"] if passed else C["fail"]
        )
        symbol = "✓" if passed else "!"

        st.markdown(
            f"""
            <div class="result-card {result_class}">
              <div class="result-label">LATEST MODEL OUTPUT</div>
              <div class="result-value" style="color:{result_color}">
                {symbol} &nbsp; {latest["prediction"]}
              </div>
              <div class="result-desc">
                <b>{latest["student"]}</b> ·
                {latest["study_hours"]:g} study hours/day
                <br>
                Generated {latest["created_at"]}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "＋ &nbsp;Start another prediction",
            use_container_width=True,
            key="new_prediction_result",
        ):
            new_prediction()

    else:
        st.markdown(
            """
            <div class="result-card"
                 style="background:var(--surface2);
                        border-color:var(--line)">
              <div class="result-label">YOUR NEXT RESULT</div>
              <div class="result-value"
                   style="color:var(--muted)">
                Waiting for input
              </div>
              <div class="result-desc">
                Complete the form and run a prediction
                to see the model output here.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Model response chart
# --------------------------------------------------
with right:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">◈ &nbsp; Model response map</div>
          <div class="panel-copy">
            Explore how the saved model classifies study-hour
            values from 0 to 24.
          </div>
        """,
        unsafe_allow_html=True,
    )

    hours_values = [
        x / 2 for x in range(0, 49)
    ]

    predicted_classes = [
        int(model.predict([[float(hours)]])[0])
        for hours in hours_values
    ]

    chart_data = pd.DataFrame(
        {
            "Study hours/day": hours_values,
            "Predicted class": predicted_classes,
        }
    ).set_index("Study hours/day")

    st.line_chart(
        chart_data,
        height=250,
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="panel-copy" style="margin-top: 14px; margin-bottom: 0;">
          0 = Fail · 1 = Pass. This chart displays predicted
          classes, not probabilities or confidence scores.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Prediction history and model information
# --------------------------------------------------
st.write("")

st.markdown(
    """
    <div class="section-heading">Activity & insights</div>
    <div class="section-copy">
        Review this session's prediction history or learn
        about the model.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_history, tab_model = st.tabs(
    ["Prediction history", "About the model"]
)

with tab_history:
    history = st.session_state.prediction_history

    if history:
        history_df = pd.DataFrame(history).rename(
            columns={
                "student": "Student",
                "study_hours": "Study hours/day",
                "prediction": "Predicted result",
                "created_at": "Timestamp",
            }
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        export_col, clear_col, _ = st.columns(
            [1.3, 1.6, 5]
        )

        with export_col:
            st.download_button(
                "↓ &nbsp;Export CSV",
                data=csv_data,
                file_name="student_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with clear_col:
            if st.button(
                "Clear session history",
                use_container_width=True,
            ):
                st.session_state.prediction_history = []
                st.session_state.pop(
                    "latest_prediction", None
                )
                st.rerun()

    else:
        st.info(
            "No predictions yet. Your results will appear "
            "here after the first prediction."
        )

with tab_model:
    info_left, info_right = st.columns(2, gap="large")

    with info_left:
        st.markdown("**What it does**")
        st.write(
            "Loads `student_pass_model.joblib` and predicts "
            "a pass/fail class using average daily study hours."
        )

        st.markdown("**Current model input**")
        st.write(
            "Study hours per day (0–24). Attendance and other "
            "academic factors are not included."
        )

    with info_right:
        st.markdown("**How to read the result**")
        st.write(
            "Pass/Fail is the model's predicted class, not "
            "a guarantee of actual academic performance."
        )

        st.markdown("**Model persistence**")
        st.write(
            "The trained model is loaded from disk. It is "
            "not retrained each time you make a prediction."
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div class="footer-note">
      EDUCATIONAL DEMO ONLY · Model outputs depend on the
      training data and model quality. Validate with
      representative data before using predictions to support
      real academic decisions.
    </div>
    """,
    unsafe_allow_html=True,
)