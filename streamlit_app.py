"""Exercise 3 — Streamlit Interface (calls the Flask API from Exercise 2)."""
import streamlit as st
import requests
import pandas as pd
import altair as alt

API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon=":material/school:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---- inline SVG icons (no emojis) ----
ICON_CAP = (
    '<svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="white" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M22 10 12 5 2 10l10 5 10-5z"/><path d="M6 12v5c0 1.7 2.7 3 6 3s6-1.3 6-3v-5"/>'
    '<line x1="22" y1="10" x2="22" y2="16"/></svg>'
)
ICON_OK = (
    '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="white" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M22 11.1V12a10 10 0 1 1-5.9-9.1"/><polyline points="22 4 12 14 9 11"/></svg>'
)
ICON_BOOK = (
    '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="white" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>'
)

# ---- Sun / Moon theme switch (dark default) ----
if "dark" not in st.session_state:
    st.session_state.dark = True
dark = st.session_state.dark
anim = "fade-d" if dark else "fade-l"

t_spacer, t_switch = st.columns([7, 1])
with t_switch:
    if st.button(
        ":material/light_mode:" if dark else ":material/dark_mode:",
        key="theme_btn",
        help="Switch to light mode" if dark else "Switch to dark mode",
    ):
        st.session_state.dark = not dark
        st.rerun()

if dark:
    BG = "linear-gradient(180deg, #0b1220 0%, #111c33 60%, #0e3a3a 100%)"
    CARD = "#16213a"
    INFO_BG = "#131f38"
    TEXT = "#e8eef7"
    MUTED = "#94a3b8"
    SECTION = "#7dd3fc"
    ACCENT = "#22d3ee"
    INPUT_BG = "#0f1a30"
    INPUT_BORDER = "#2b3b55"
    CODE_BG = "#0b1426"
    SCHEME = "dark"
    GRID_C = "#24334d"
    AXIS_C = "#cbd5e1"
    LINE_C = "#22d3ee"
    PT_C = "#38bdf8"
    RULE_C = "#94a3b8"
else:
    BG = "linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdfa 100%)"
    CARD = "white"
    INFO_BG = "white"
    TEXT = "#0f172a"
    MUTED = "#64748b"
    SECTION = "#0369a1"
    ACCENT = "#0369a1"
    INPUT_BG = "#ffffff"
    INPUT_BORDER = "#bae6fd"
    CODE_BG = "#f0f9ff"
    SCHEME = "light"
    GRID_C = "#e2e8f0"
    AXIS_C = "#334155"
    LINE_C = "#0ea5e9"
    PT_C = "#1f77b4"
    RULE_C = "gray"

CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background: __BG__; color: __TEXT__; }
    .stApp input { color-scheme: __SCHEME__; }
    div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:first-child div[data-testid="column"]:last-child {
        text-align: right;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:first-child div[data-testid="stButton"] button {
        width: 48px; height: 48px; border-radius: 50% !important;
        padding: 0 !important;
        background: __CARD__ !important;
        border: 2px solid __INPUT_BORDER__ !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.25);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:first-child div[data-testid="stButton"] button:hover {
        transform: rotate(25deg) scale(1.08);
        box-shadow: 0 8px 26px rgba(14, 165, 233, 0.45);
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:first-child div[data-testid="stButton"] button span {
        display: inline-block;
        animation: iconPop 0.55s ease;
        font-size: 24px;
    }
    @keyframes iconPop {
        0% { transform: rotate(-150deg) scale(0.3); opacity: 0; }
        60% { transform: rotate(18deg) scale(1.2); opacity: 1; }
        100% { transform: rotate(0deg) scale(1); opacity: 1; }
    }
    .fade-d { animation: fadeD 0.5s ease; }
    .fade-l { animation: fadeL 0.5s ease; }
    @keyframes fadeD { from { opacity: 0.2; transform: translateY(8px); } to { opacity: 1; transform: none; } }
    @keyframes fadeL { from { opacity: 0.2; transform: translateY(8px); } to { opacity: 1; transform: none; } }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 950px; }
    .hero {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 50%, #14b8a6 100%);
        padding: 3rem 2rem; border-radius: 24px; color: white;
        text-align: center; box-shadow: 0 20px 45px rgba(14, 165, 233, 0.35);
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.5rem; font-weight: 900; margin: 0.5rem 0 0 0; color: white !important; letter-spacing: -1px; }
    .hero p { font-size: 1.1rem; margin-top: 0.75rem; color: white !important; }
    .hero .badge {
        display: inline-block; background: rgba(255,255,255,0.28);
        padding: 0.4rem 1.1rem; border-radius: 999px; font-size: 0.85rem;
        margin-top: 1.2rem; color: white !important; font-weight: 700;
    }
    .section-title {
        font-size: 1.15rem; font-weight: 800; color: __SECTION__;
        margin: 0 0 1.2rem 0; text-transform: uppercase; letter-spacing: 0.8px;
    }
    .stTextInput label, .stNumberInput label, .stSlider label {
        font-weight: 700 !important; color: __TEXT__ !important; font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 12px !important; border: 2px solid __INPUT_BORDER__ !important;
        padding: 0.7rem 1rem !important; font-size: 1rem !important;
        background: __INPUT_BG__ !important; color: __TEXT__ !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        color: white !important; border: none; padding: 0.85rem 2rem;
        border-radius: 12px; font-weight: 800; font-size: 1.05rem;
        width: 100%; box-shadow: 0 8px 22px rgba(14, 165, 233, 0.4);
    }
    .result-pass {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        padding: 2rem; border-radius: 20px; color: white !important;
        text-align: center; box-shadow: 0 15px 40px rgba(16, 185, 129, 0.35);
        margin-bottom: 1.5rem;
    }
    .result-fail {
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        padding: 2rem; border-radius: 20px; color: white !important;
        text-align: center; box-shadow: 0 15px 40px rgba(239, 68, 68, 0.35);
        margin-bottom: 1.5rem;
    }
    .result-pass h2, .result-fail h2 {
        margin: 0.4rem 0 0 0; font-size: 1.9rem; font-weight: 900; color: white !important;
    }
    .result-pass p, .result-fail p {
        margin: 0.6rem 0 0 0; font-size: 1rem; color: white !important;
    }
    .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: __CARD__; padding: 1.4rem 1rem; border-radius: 16px;
        text-align: center; box-shadow: 0 4px 20px rgba(14, 165, 233, 0.12);
        border: 2px solid __INPUT_BORDER__;
    }
    .metric-label {
        font-size: 0.75rem; color: __MUTED__; text-transform: uppercase;
        letter-spacing: 1.3px; font-weight: 800;
    }
    .metric-value { font-size: 1.8rem; font-weight: 900; color: __ACCENT__; margin-top: 0.4rem; }
    .info-box {
        background: __INFO_BG__; border-left: 5px solid #0ea5e9;
        padding: 1.3rem 1.5rem; border-radius: 14px; margin-top: 1rem;
        color: __TEXT__; font-size: 0.95rem; line-height: 1.7;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.1);
    }
    .info-box strong { color: __ACCENT__; }
    .info-box code {
        background: __CODE_BG__; padding: 0.15rem 0.45rem; border-radius: 5px;
        color: #db2777; font-size: 0.85rem; border: 1px solid __INPUT_BORDER__;
    }
    .placeholder {
        background: __CARD__; padding: 3.5rem 2rem; border-radius: 20px;
        border: 2px dashed __SECTION__; text-align: center; color: __MUTED__;
    }
    .placeholder p { font-size: 1.05rem; margin: 0.8rem 0 0 0; color: __MUTED__; }
    .placeholder strong { color: __ACCENT__; }
    .footer {
        text-align: center; color: __MUTED__; font-size: 0.85rem;
        margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid __INPUT_BORDER__;
    }
    .footer strong { color: __ACCENT__; }
</style>
"""
for token, val in [("__BG__", BG), ("__CARD__", CARD), ("__INFO_BG__", INFO_BG),
                   ("__TEXT__", TEXT), ("__MUTED__", MUTED), ("__SECTION__", SECTION),
                   ("__ACCENT__", ACCENT), ("__INPUT_BG__", INPUT_BG),
                   ("__INPUT_BORDER__", INPUT_BORDER), ("__CODE_BG__", CODE_BG),
                   ("__SCHEME__", SCHEME)]:
    CSS = CSS.replace(token, val)
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(f"""
<div class="hero {anim}">
    {ICON_CAP}
    <h1>Student Pass Predictor</h1>
    <p>AI-powered prediction using Machine Learning</p>
    <div class="badge">Flask API + Logistic Regression</div>
</div>
""", unsafe_allow_html=True)

st.markdown(':material/edit: **STUDENT INFORMATION**', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    student_name = st.text_input(
        ":material/person: Student Name",
        placeholder="Enter Name...",
        key="name_input",
    )
with col2:
    study_hours = st.number_input(
        ":material/book: Study Hours per Day",
        min_value=0.0, max_value=24.0, value=5.0, step=0.5,
    )

attendance = st.slider(
    ":material/calendar_check: Attendance Percentage",
    min_value=0, max_value=100, value=85,
)

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button(":material/auto_awesome: Predict Result", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

if predict_clicked:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        payload = {
            "student_name": student_name.strip(),
            "study_hours": study_hours,
            "attendance_percentage": attendance,
        }
        try:
            with st.spinner("Contacting Flask API..."):
                response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                label = result.get("prediction_label") or result.get("result", "")
                passed = str(label).lower() == "pass"
                icon = ICON_OK if passed else ICON_BOOK
                css = "result-pass" if passed else "result-fail"
                verb = "pass" if passed else "fail"
                display_name = result.get("student_name") or student_name.strip()

                st.markdown(f"""
                <div class="{css}">
                    {icon}
                    <h2>{display_name} — {label.upper()}</h2>
                    <p>This student is predicted to {verb} the course.</p>
                </div>
                """, unsafe_allow_html=True)

                conf = result.get("confidence")
                conf_text = f"{conf}%" if conf is not None else "n/a"
                st.markdown(f"""
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Prediction</div>
                        <div class="metric-value">{label}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{conf_text}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Study Hours</div>
                        <div class="metric-value">{result['study_hours']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-box">
                    <strong>How this was generated:</strong><br>
                    Streamlit sent a POST request to the Flask API with
                    <strong>{result['study_hours']} study hours</strong> and
                    <strong>{result['attendance_percentage']}% attendance</strong>.
                    The backend loaded <code>student_pass_model.joblib</code> and
                    returned <strong>{label}</strong>
                    ({conf_text} confidence). Both study hours and attendance
                    shape the prediction.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(':material/analytics: **STUDY HOURS VS PREDICTED CLASS**')

                # Decision curve via the API (one call per hour mark)
                x_max = max(11, float(study_hours) + 1)
                rows = []
                h = 0.0
                while h <= x_max:
                    try:
                        r = requests.post(API_URL, json={
                            "study_hours": round(h, 1),
                            "attendance_percentage": attendance,
                        }, timeout=5)
                        d = r.json()
                        lab = d.get("prediction_label") or d.get("result", "")
                        rows.append({"Study hours": round(h, 1),
                                     "Class": 1 if str(lab).lower() == "pass" else 0})
                    except Exception:
                        rows.append({"Study hours": round(h, 1), "Class": None})
                    h = round(h + 1.0, 1)
                curve_df = pd.DataFrame(rows)

                line = alt.Chart(curve_df).mark_line(color=LINE_C, strokeWidth=3).encode(
                    x=alt.X("Study hours:Q", title="Study hours per day",
                            scale=alt.Scale(domain=[0, x_max])),
                    y=alt.Y("Class:Q", title="Predicted class (0 = FAIL, 1 = PASS)",
                            scale=alt.Scale(domain=[0, 1.3])),
                )
                you_pt = alt.Chart(pd.DataFrame([{
                    "Study hours": float(study_hours),
                    "Class": 1 if passed else 0,
                }])).mark_point(size=220, shape="diamond", filled=True, color=PT_C).encode(
                    x=alt.X("Study hours:Q", scale=alt.Scale(domain=[0, x_max])),
                    y="Class:Q",
                )
                boundary = alt.Chart(pd.DataFrame({"x": [5.5]})).mark_rule(
                    strokeDash=[5, 4], color=RULE_C).encode(
                    x=alt.X("x:Q", scale=alt.Scale(domain=[0, x_max])))
                themed = (line + you_pt + boundary).properties(height=320).configure_axis(
                    labelColor=AXIS_C, titleColor=AXIS_C, gridColor=GRID_C
                ).configure_view(fill=CARD, strokeWidth=0)
                st.altair_chart(themed, use_container_width=True)
                st.caption(
                    f"Blue diamond = {display_name} at {study_hours} hrs "
                    f"→ {label}."
                )
            else:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                st.error(f"API Error {response.status_code}: {detail}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Flask API. Make sure `python app.py` is running on port 5000.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
else:
    st.markdown("""
    <div class="placeholder">
        <p>Fill in the student information above and click<br>
        <strong>Predict Result</strong> to see the prediction.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <strong>IT Elective 3 — AI: Machine Learning</strong><br>
    Model Persistence & Deployment<br>
    Frontend: Streamlit · Backend: Flask API · Model: Logistic Regression
</div>
""", unsafe_allow_html=True)
