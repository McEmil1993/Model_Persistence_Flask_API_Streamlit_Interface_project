"""Exercise 3 — Student Pass Prediction dashboard.

Run: streamlit run streamlit_app.py

"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Classroom | Student Insights", page_icon="📘", layout="wide", initial_sidebar_state="expanded")

# Dark analytics dashboard: deep navy surfaces, blue/cyan accents, light input fields with dark text.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --ink:#e8eef8; --muted:#9aabc4; --bg:#08111f; --panel:#111e31; --panel2:#16263d;
  --blue:#3984ff; --blue2:#78adff; --line:#263852; --cyan:#55d6e8;
  --field-bg:#f4f7fc; --field-border:#b8c7da; --field-text:#0f172a; --field-placeholder:#64748b;
}

/* ---------- Base ---------- */
.stApp { background:radial-gradient(ellipse at 85% 0%, #142b4c 0%, #08111f 42%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(8,17,31,.88); }
.block-container { max-width:1440px; padding-top:1.6rem; padding-bottom:3rem; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink) !important; letter-spacing:-.035em; }
h1 { font-size:2.55rem !important; font-weight:800 !important; }
h2 { font-weight:750 !important; }
h3 { font-size:1.35rem !important; font-weight:700 !important; }
p, label, .stMarkdown { color:var(--ink); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background:#0b1728; border-right:1px solid #223652; }
[data-testid="stSidebar"] * { color:#e8eef8 !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#fff !important; }
.brand-mark { width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,#3984ff,#55d6e8);display:inline-flex;align-items:center;justify-content:center;color:#071426 !important;font-size:23px;font-weight:900;margin-bottom:18px;box-shadow:0 8px 26px #3984ff30; }
[data-testid="stSidebar"] .brand-mark { color:#071426 !important; }
.sidebar-kicker { color:#78adff !important;font-size:.7rem;letter-spacing:.18em;font-weight:700;text-transform:uppercase; }
.sidebar-copy { color:#b6c5da !important;font-size:.9rem;line-height:1.65; }

/* ---------- Headings and copy ---------- */
.eyebrow { font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; color:var(--cyan); }
.lead { color:var(--muted) !important; font-size:1.05rem; margin-top:-.45rem; max-width:62ch; }

/* ---------- Cards ---------- */
.panel { background:linear-gradient(145deg,#14243a,#101b2d); border:1px solid var(--line); border-radius:20px; padding:1.2rem 1.35rem; height:100%; box-shadow:0 12px 32px #00000018; }
.panel-label { color:var(--muted); text-transform:uppercase; letter-spacing:.1em; font-size:.7rem; font-weight:800; }
.panel-value { font-family:'Manrope',sans-serif; font-size:1.65rem; line-height:1.25; font-weight:800; color:var(--ink); margin-top:.45rem; }
.panel-note { color:var(--muted); font-size:.84rem; margin-top:.35rem; }

.result-pass { background:linear-gradient(135deg,#103b43,#112b3c); border:1px solid #247b87; border-radius:20px; padding:1.35rem 1.5rem; margin-bottom:1rem; }
.result-fail { background:linear-gradient(135deg,#3b2535,#2a2032); border:1px solid #87506b; border-radius:20px; padding:1.35rem 1.5rem; margin-bottom:1rem; }
.result-pass, .result-fail { color:var(--ink); }
.result-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.15em; font-weight:800; color:#b7c8df; }
.result-word { font-family:'Manrope',sans-serif; font-size:2.6rem; font-weight:800; line-height:1.2; margin:.35rem 0; color:#fff; }
.result-pass .result-word { color:#7ff0d0; }
.result-fail .result-word { color:#ff9db8; }

.note { background:#14243a; border:1px solid #263852; border-left:4px solid var(--cyan); padding:.85rem 1rem; border-radius:0 12px 12px 0; color:#c3d0e2; font-size:.86rem; line-height:1.55; }

/* ---------- Form container ---------- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] [data-testid="stTextInput"]) {
  background:linear-gradient(145deg,#14243a,#101b2d);
  border:1px solid var(--line) !important;
  border-radius:20px !important;
  padding:.6rem .5rem;
  box-shadow:0 12px 32px #00000018;
}

/* ---------- Field labels ---------- */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
  color:var(--ink) !important; font-weight:600 !important; font-size:.92rem !important; letter-spacing:.01em;
}
[data-testid="stTooltipIcon"] svg { color:var(--muted) !important; }

/* ---------- Inputs: light surface + dark text (text AND number fields) ---------- */
div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="base-input"],
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
  background:var(--field-bg) !important;
  border-color:var(--field-border) !important;
  border-radius:10px !important;
}
div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[data-baseweb="input"] {
  border:1px solid var(--field-border) !important;
  transition:border-color .15s ease, box-shadow .15s ease;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  color:var(--field-text) !important;
  -webkit-text-fill-color:var(--field-text) !important;
  caret-color:var(--field-text) !important;
  font-weight:500;
  min-height:2.6rem;
}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stNumberInput"] input::placeholder {
  color:var(--field-placeholder) !important;
  -webkit-text-fill-color:var(--field-placeholder) !important;
  opacity:1 !important;
}
div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
  border-color:var(--blue) !important;
  box-shadow:0 0 0 3px #3984ff40 !important;
}
/* Number stepper buttons */
[data-testid="stNumberInput"] button { color:#1e3a8a !important; background:#e3eaf5 !important; border:none !important; }
[data-testid="stNumberInput"] button:hover { background:#d1ddf0 !important; color:#1e40af !important; }
/* Browser autofill should not turn the field white-on-white or yellow */
div[data-testid="stTextInput"] input:-webkit-autofill {
  -webkit-text-fill-color:var(--field-text) !important;
  box-shadow:0 0 0 1000px var(--field-bg) inset !important;
}

/* ---------- Button ---------- */
div.stButton > button { background:linear-gradient(100deg,#2868df,#3984ff); color:#fff; border:1px solid #3984ff; border-radius:11px; min-height:3rem; width:100%; font-weight:800; letter-spacing:.01em; transition:all .2s ease; box-shadow:0 8px 22px #3984ff22; }
div.stButton > button:hover { background:linear-gradient(100deg,#1f56bd,#2f73e8); color:#fff; border-color:#78adff; transform:translateY(-1px); }
div.stButton > button:active { transform:translateY(0); }
div.stButton > button:focus-visible { outline:3px solid #78adff66; outline-offset:2px; }

/* ---------- Data display ---------- */
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#3984ff,#55d6e8); }
[data-testid="stProgress"] p, [data-testid="stProgress"] label { color:var(--muted) !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stMetric"] { background:#14243a; border:1px solid #263852; padding:12px 14px; border-radius:14px; }
[data-testid="stMetricLabel"] { color:#9aabc4 !important; }
[data-testid="stMetricValue"] { color:#e8eef8 !important; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; background:#0f1c2f; }
[data-testid="stExpander"] summary p { color:var(--ink) !important; font-weight:600; }
[data-testid="stCaptionContainer"], .stCaption { color:var(--muted) !important; }
.stAlert { border-radius:12px; }
hr { border-color:var(--line); }
footer { visibility:hidden; }

@media (max-width:700px) { h1 {font-size:2rem !important;} .block-container {padding-top:1rem;} .result-word {font-size:2.1rem;} }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="brand-mark">✳</div><div class="sidebar-kicker">Classroom / Lab 01</div>', unsafe_allow_html=True)
    st.markdown("## Student Insights")
    st.markdown('<p class="sidebar-copy">A compact learning dashboard for exploring how study time relates to a sample pass/fail model.</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<p class="sidebar-kicker">MODEL STATUS</p>', unsafe_allow_html=True)
    if MODEL_PATH.exists():
        st.success("Saved model connected")
        st.caption("Logistic regression · local model file")
    else:
        st.warning("Model file not found")
        st.code("python train_model.py", language="bash")
    st.divider()
    st.markdown('<p class="sidebar-kicker">HOW TO USE</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-copy">1. Add a student name<br>2. Enter study hours and attendance<br>3. Generate an estimate<br>4. Review the chart</p>', unsafe_allow_html=True)
    st.caption("Educational prototype · not an official assessment")


st.markdown('<div class="eyebrow">LEARNING ANALYTICS &nbsp; / &nbsp; EXERCISE 03</div>', unsafe_allow_html=True)
st.title("Student performance, made clearer.")
st.markdown('<p class="lead">Explore a sample machine-learning prediction with a dark, modern analytics workspace.</p>', unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("The trained model has not been created yet. Train it first, then refresh this page.")
    st.code("python train_model.py", language="bash")
    st.stop()

model = joblib.load(MODEL_PATH)

# Top overview cards
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown('<div class="panel"><div class="panel-label">Prediction engine</div><div class="panel-value">Ready to use</div><div class="panel-note">Saved model loaded locally</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="panel"><div class="panel-label">Main input</div><div class="panel-value">Study hours</div><div class="panel-note">The only feature used by this model</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="panel"><div class="panel-label">Input checks</div><div class="panel-value">0–24 h · 0–100%</div><div class="panel-note">Study time and attendance ranges</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([0.85, 1.15], gap="large")
with left:
    st.markdown("### Student record")
    st.markdown("Use the fields below to create a prediction record.")
    with st.container(border=True):
        student_name = st.text_input("Student name", placeholder="e.g. Alex Rivera")
        study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=5.0, step=0.5, help="Enter a value between 0 and 24 hours.")
        attendance = st.number_input("Attendance percentage", min_value=0.0, max_value=100.0, value=85.0, step=1.0, help="Enter a value between 0 and 100%. Attendance is recorded but is not used by the current model.")
        st.markdown('<div class="note">Attendance is included for the activity and validated from 0–100%. The current trained model uses study hours only, so attendance does not change the prediction.</div>', unsafe_allow_html=True)
        st.write("")
        predict_clicked = st.button("Generate student estimate  →", use_container_width=True)

with right:
    st.markdown("### Prediction overview")
    if predict_clicked:
        prediction = int(model.predict([[study_hours]])[0])
        label = "PASS" if prediction == 1 else "FAIL"
        probability = float(model.predict_proba([[study_hours]])[0][1])
        display_name = student_name.strip() or "Unnamed student"
        result_class = "result-pass" if prediction == 1 else "result-fail"
        st.markdown(f'<div class="{result_class}"><div class="result-label">Model estimate for {display_name}</div><div class="result-word">{label}</div><div>Estimated pass probability: <b>{probability:.0%}</b></div></div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Study time", f"{study_hours:g} hrs/day")
        with m2:
            st.metric("Attendance recorded", f"{attendance:.0f}%")
        st.progress(probability, text="Model-estimated pass probability")
        st.caption("Probability is a model output, not a guarantee of academic performance.")
    else:
        st.markdown('<div class="panel"><div class="panel-label">Your result will appear here</div><div class="panel-value">Ready when you are.</div><div class="panel-note">Enter the student details, then generate an estimate.</div></div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="note">Tip: try several study-hour values to see how the model output changes. Use this as a learning demonstration, not a real academic decision.</div>', unsafe_allow_html=True)

st.divider()
st.markdown("### Explore the model")
st.markdown("Compare the estimated pass probability across study-hour values. The dashed line marks your current input.")
hours_range = np.linspace(0, 10, 101).reshape(-1, 1)
pass_probabilities = model.predict_proba(hours_range)[:, 1]
fig, ax = plt.subplots(figsize=(12, 4.2))
fig.patch.set_facecolor("#111e31")
ax.set_facecolor("#111e31")
ax.plot(hours_range.ravel(), pass_probabilities, color="#3984ff", linewidth=3, label="Estimated pass probability")
ax.fill_between(hours_range.ravel(), pass_probabilities, alpha=.10, color="#3984ff")
ax.axhline(0.5, color="#9aabc4", linestyle=":", linewidth=1.2, alpha=.7, label="50% pass threshold")
ax.axvline(study_hours, color="#55d6e8", linestyle="--", linewidth=2, label=f"Current input: {study_hours:g} h/day")
ax.set_xlabel("Study hours per day", color="#b6c5da", labelpad=10)
ax.set_ylabel("Estimated probability", color="#b6c5da", labelpad=10)
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
ax.grid(True, color="#263852", linewidth=.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#3a4f6e")
ax.spines["bottom"].set_color("#3a4f6e")
ax.tick_params(colors="#9aabc4")
ax.legend(frameon=False, loc="lower right", labelcolor="#c3d0e2")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

with st.expander("View sample comparison table"):
    sample_hours = np.array([0, 2, 4, 6, 8, 10]).reshape(-1, 1)
    sample_probabilities = model.predict_proba(sample_hours)[:, 1]
    comparison = pd.DataFrame({"Study hours/day": sample_hours.ravel(), "Estimated pass probability": [f"{p:.0%}" for p in sample_probabilities], "Model label": ["Pass" if p >= .5 else "Fail" for p in sample_probabilities]})
    st.dataframe(comparison, hide_index=True, use_container_width=True)

st.divider()
st.caption("STUDENT INSIGHTS  ·  Python / scikit-learn / Streamlit  ·  Sample data for educational use")