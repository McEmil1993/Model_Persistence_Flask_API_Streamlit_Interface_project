import html
import time

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILE = "student_pass_model.joblib"

BG, CARD, INK = "#F6F7F9", "#FFFFFF", "#111827"
ACCENT, MUTED, BORDER = "#1F3A5F", "#6B7280", "#E5E7EB"
GREEN, RED = "#15803D", "#B91C1C"

FRAMES = 16
FRAME_DELAY = 0.035

st.set_page_config(page_title="Pass / Fail?", page_icon="🎯", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {BG}; color: {INK}; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding: 2.5rem 3rem 2rem; max-width: 1280px; }}

.topbar {{ border-bottom: 1px solid {BORDER}; padding-bottom: 1.1rem; margin-bottom: 1.8rem; }}
.topbar h1 {{
  font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; color: {INK};
}}
.topbar p {{ color: {MUTED}; margin: .35rem 0 0; font-size: .98rem; }}

.st-key-form, .st-key-result, .st-key-chart, .st-key-how {{
  background: {CARD} !important; border: 1px solid {BORDER} !important; border-radius: 10px !important;
  box-shadow: 0 1px 2px rgba(17, 24, 39, .04); padding: 1.2rem 1.3rem !important;
}}
.st-key-chart {{ min-height: 600px; }}
h3.panel {{
  font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  margin: 0 0 1rem; color: {MUTED};
}}

label p {{ color: {INK} !important; font-weight: 500; font-size: .92rem; }}
div[data-baseweb="input"], div[data-baseweb="base-input"] {{
  background: {CARD} !important; border-radius: 8px !important;
}}
div[data-baseweb="input"] {{ border: 1px solid #D1D5DB !important; }}
div[data-baseweb="input"]:focus-within {{ box-shadow: 0 0 0 3px {ACCENT}22 !important; border-color: {ACCENT} !important; }}
.stTextInput input {{ color: {INK} !important; background: {CARD} !important; }}
.stSlider div[role="slider"] {{ background: {ACCENT} !important; border: 2px solid #fff; box-shadow: 0 0 0 1px {ACCENT}; }}
.stSlider [data-testid="stSliderThumbValue"] {{ color: {INK}; font-weight: 500; }}
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {{ color: {MUTED}; }}

.stButton > button {{
  width: 100%; background: {ACCENT}; color: #fff; border: 1px solid {ACCENT}; border-radius: 8px;
  padding: .6rem 1rem; font-weight: 600; font-size: .95rem; transition: background .15s ease;
}}
.stButton > button:hover {{ background: #16304F; color: #fff; border-color: #16304F; }}
.stButton > button:active {{ background: #0F2440; color: #fff; }}
.stButton > button:focus-visible {{ outline: 3px solid {ACCENT}44; outline-offset: 2px; }}

.verdict-box {{ border: 1px solid {BORDER}; border-left: 5px solid {INK}; border-radius: 8px; padding: 1rem 1.1rem; }}
.verdict-box.pass {{ border-left-color: {GREEN}; }}
.verdict-box.fail {{ border-left-color: {RED}; }}
.verdict-box .verdict {{ font-size: 2rem; font-weight: 700; line-height: 1.1; letter-spacing: -0.02em; }}
.verdict-box.pass .verdict {{ color: {GREEN}; }}
.verdict-box.fail .verdict {{ color: {RED}; }}
.verdict-box .who {{ color: {INK}; font-weight: 500; margin-top: .45rem; }}
.verdict-box .prob {{ color: {MUTED}; font-size: .93rem; margin-top: .1rem; }}
.empty {{ border: 1px dashed #D1D5DB; border-radius: 8px; padding: 1.1rem; color: {MUTED}; font-size: .95rem; }}

.step {{ display: flex; gap: .8rem; align-items: flex-start; }}
.step .num {{
  flex: 0 0 1.7rem; height: 1.7rem; border-radius: 50%; background: {ACCENT}; color: #fff;
  display: flex; align-items: center; justify-content: center; font-size: .82rem; font-weight: 600;
}}
.step .text {{ color: {INK}; font-size: .95rem; line-height: 1.5; }}
.step code {{ background: #EEF1F5; color: {INK}; border-radius: 4px; padding: .05rem .3rem; font-size: .85rem; }}
.note {{ color: {MUTED}; font-size: .88rem; margin-top: .6rem; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


model = load_model()

st.markdown(
    '<div class="topbar"><h1>Pass or fail?</h1>'
    '<p>Enter a student and their study hours, then press Predict result.</p></div>',
    unsafe_allow_html=True,
)

GRID = np.linspace(0, 10, 101)
CURVE = model.predict_proba(GRID.reshape(-1, 1))[:, 1]


def make_chart(probs, result=None):
    axis = dict(labelColor=MUTED, titleColor=INK, titleFontWeight=500, gridColor="#EEF0F3",
                domainColor="#9CA3AF", tickColor="#9CA3AF", domainWidth=1)
    df = pd.DataFrame({"hours": GRID, "prob": probs})
    base = alt.Chart(df).encode(
        x=alt.X("hours:Q", title="Study hours", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(**axis)),
        y=alt.Y("prob:Q", title="Probability of passing", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", **axis)),
    )
    layers = [
        base.mark_line(color=ACCENT, strokeWidth=3),
        alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=MUTED, strokeDash=[6, 5]).encode(y="y:Q"),
        alt.Chart(pd.DataFrame({"x": [0.2], "y": [0.54], "t": ["50% line: pass above, fail below"]}))
        .mark_text(color=MUTED, align="left", fontSize=12, fontWeight=500).encode(x="x:Q", y="y:Q", text="t:N"),
    ]
    if result:
        color = GREEN if result["pred"] == 1 else RED
        pt = pd.DataFrame({"hours": [result["hours"]], "prob": [result["prob"]], "student": [result["name"]]})
        layers.append(alt.Chart(pt).mark_rule(color=color, strokeWidth=1.5, strokeDash=[3, 4]).encode(x="hours:Q"))
        layers.append(alt.Chart(pt).mark_point(filled=True, size=200, color=color, stroke="#fff", strokeWidth=2)
                      .encode(x="hours:Q", y="prob:Q",
                              tooltip=["student:N", alt.Tooltip("hours:Q", format=".1f"),
                                       alt.Tooltip("prob:Q", format=".0%")]))
    return (alt.layer(*layers).properties(height=460)
            .configure(background="transparent").configure_view(strokeWidth=0))


side, main = st.columns([1, 2.1], gap="large")

with side:
    with st.container(border=True, key="form"):
        st.markdown('<h3 class="panel">Student details</h3>', unsafe_allow_html=True)
        name = st.text_input("Student name", placeholder="e.g. Aisha")
        hours = st.slider("Study hours", 0.0, 10.0, 5.0, 0.5)
        st.write("")
        clicked = st.button("Predict result")

animate = False
if clicked:
    if not name.strip():
        st.session_state.pop("result", None)
        st.session_state["missing_name"] = True
    else:
        st.session_state["missing_name"] = False
        pred = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])
        st.session_state["result"] = dict(name=name.strip(), hours=hours, pred=pred, prob=prob)
        animate = True

result = st.session_state.get("result")

with side:
    st.write("")
    with st.container(border=True, key="result"):
        st.markdown('<h3 class="panel">Result</h3>', unsafe_allow_html=True)
        if st.session_state.get("missing_name"):
            st.warning("Enter the student's name, then press Predict result.")
        if result:
            ok = result["pred"] == 1
            st.markdown(
                f"""<div class="verdict-box {'pass' if ok else 'fail'}">
                <div class="verdict">{'PASS' if ok else 'FAIL'}</div>
                <div class="who">{html.escape(result['name'])}, {result['hours']:g} study hours</div>
                <div class="prob">{result['prob']:.0%} chance of passing</div></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">No prediction yet.</div>', unsafe_allow_html=True)

with main:
    with st.container(border=True, key="chart"):
        st.markdown('<h3 class="panel">Study hours vs predicted class</h3>', unsafe_allow_html=True)
        slot = st.empty()
        flat = np.zeros_like(CURVE)

        if animate:
            for i in range(1, FRAMES + 1):
                t = 1 - (1 - i / FRAMES) ** 3
                slot.altair_chart(make_chart(flat + t * (CURVE - flat)), width="stretch")
                time.sleep(FRAME_DELAY)
            slot.altair_chart(make_chart(CURVE, result), width="stretch")
        elif result:
            slot.altair_chart(make_chart(CURVE, result), width="stretch")
        else:
            slot.altair_chart(make_chart(flat), width="stretch")

        st.markdown(
            '<div class="note">The line stays flat until you press Predict result. '
            'Then it draws the model\'s curve and marks the student.</div>',
            unsafe_allow_html=True)

st.write("")
with st.container(border=True, key="how"):
    st.markdown('<h3 class="panel">How it works</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    c1.markdown(
        f'<div class="step"><div class="num">1</div><div class="text">The app loads the saved model '
        f'<code>{MODEL_FILE}</code>. It is not retrained.</div></div>',
        unsafe_allow_html=True)
    c2.markdown(
        '<div class="step"><div class="num">2</div><div class="text">The model turns the study hours '
        'into a probability of passing.</div></div>',
        unsafe_allow_html=True)
    c3.markdown(
        '<div class="step"><div class="num">3</div><div class="text">Above 50% the class is PASS. '
        'At or below 50% it is FAIL.</div></div>',
        unsafe_allow_html=True)