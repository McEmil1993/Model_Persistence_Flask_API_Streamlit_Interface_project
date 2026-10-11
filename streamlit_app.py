import html

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILE = "student_pass_model.joblib"  

INK, PANEL, CHALK, MUTED = "#14122B", "#1E1A44", "#F3EEFF", "#A9A2D4"
GOLD, MINT, CORAL = "#FFB627", "#3DDC97", "#FF5D73"

st.set_page_config(page_title="Pass Predictor", page_icon="🎓", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Instrument Sans', sans-serif; }}
.stApp {{
  background: radial-gradient(1200px 600px at 85% -10%, #2D2670 0%, transparent 60%), {INK};
  color: {CHALK};
}}
#MainMenu, footer, header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding: 2rem 2.5rem 1.5rem; max-width: 1500px; }}

h1.title {{
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: 2.6rem; line-height: 1.05; margin: 0 0 .25rem; color: {CHALK};
  letter-spacing: -0.02em;
}}
p.sub {{ color: {MUTED}; margin: 0 0 1.4rem; font-size: 1.05rem; }}

/* panels */
.st-key-p1, .st-key-p2, .st-key-p3 {{
  background: {PANEL} !important; border: 1px solid #3A3478 !important; border-radius: 22px !important;
  padding: 1rem 1.1rem !important; box-shadow: 0 18px 40px -22px #000; min-height: 560px;
}}
div[data-baseweb="input"], div[data-baseweb="base-input"] {{
  background: {INK} !important; border-color: #3A3478 !important; border-radius: 12px !important;
}}
div[data-baseweb="input"]:focus-within {{ border-color: {GOLD} !important; box-shadow: 0 0 0 1px {GOLD} !important; }}
h3.panel {{
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 500;
  font-size: 1.15rem; margin: 0 0 .8rem; color: {CHALK};
}}

/* inputs */
label, .stTextInput label p, .stNumberInput label p, .stSlider label p {{ color: {MUTED} !important; font-weight: 500; }}
.stTextInput input, .stNumberInput input {{
  background: {INK} !important; color: {CHALK} !important;
  border: 1px solid #3A3478 !important; border-radius: 12px !important;
}}
div[data-testid="stNumberInput"] button {{ background: {INK}; color: {CHALK}; border-color: #3A3478; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {GOLD}; box-shadow: none; }}
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {{ color: {MUTED}; }}

.stButton > button {{
  width: 100%; background: {GOLD}; color: {INK}; border: 0; border-radius: 14px;
  padding: .75rem 1rem; font-weight: 600; font-size: 1.05rem;
  font-family: 'Bricolage Grotesque', sans-serif;
  transition: transform .12s ease, box-shadow .12s ease;
}}
.stButton > button:hover {{ background: #FFC24D; color: {INK}; transform: translateY(-2px); box-shadow: 0 10px 24px -10px {GOLD}; }}
.stButton > button:focus-visible {{ outline: 3px solid {CHALK}; outline-offset: 2px; }}

/* result stamp */
.stamp {{ border-radius: 18px; padding: 1.1rem 1.3rem; border: 2px solid; margin-bottom: 1rem; }}
.stamp.pass {{ border-color: {MINT}; background: rgba(61,220,151,.10); }}
.stamp.fail {{ border-color: {CORAL}; background: rgba(255,93,115,.10); }}
.stamp .verdict {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 3rem; line-height: 1; }}
.stamp.pass .verdict {{ color: {MINT}; }}
.stamp.fail .verdict {{ color: {CORAL}; }}
.stamp .who {{ font-size: 1.05rem; margin-top: .45rem; color: {CHALK}; }}
.stamp .prob {{ color: {MUTED}; margin-top: .15rem; }}
.empty {{ border: 2px dashed #3A3478; border-radius: 18px; padding: 1.4rem 1.3rem; color: {MUTED}; margin-bottom: 1rem; }}

ol.how {{ margin: 0; padding-left: 1.2rem; color: {CHALK}; line-height: 1.55; }}
ol.how li {{ margin-bottom: .45rem; }}
ol.how li::marker {{ color: {GOLD}; font-weight: 600; }}
.note {{ color: {MUTED}; font-size: .92rem; margin-top: .6rem; }}
div[data-testid="stAlert"] {{ border-radius: 12px; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


model = load_model()

st.markdown('<h1 class="title">Will they pass?</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Enter a student and their study hours. The model answers instantly.</p>',
    unsafe_allow_html=True,
)

left, middle, right = st.columns([1, 1.1, 1.6], gap="medium")

with left:
    with st.container(border=True, key="p1"):
        st.markdown('<h3 class="panel">Student</h3>', unsafe_allow_html=True)
        name = st.text_input("Student name", placeholder="e.g. Aisha")
        hours = st.number_input("Study hours", min_value=0.0, max_value=10.0,
                                value=5.0, step=0.5)
        st.write("")
        clicked = st.button("Predict result")

if clicked:
    if not name.strip():
        st.session_state.pop("result", None)
        st.session_state["missing_name"] = True
    else:
        st.session_state["missing_name"] = False
        pred = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])
        st.session_state["result"] = dict(name=name.strip(), hours=hours,
                                          pred=pred, prob=prob)

result = st.session_state.get("result")

with middle:
    with st.container(border=True, key="p2"):
        st.markdown('<h3 class="panel">Result</h3>', unsafe_allow_html=True)
        if st.session_state.get("missing_name"):
            st.warning("Enter the student's name, then press Predict result.")
        if result:
            ok = result["pred"] == 1
            st.markdown(
                f"""<div class="stamp {'pass' if ok else 'fail'}">
                <div class="verdict">{'PASS' if ok else 'FAIL'}</div>
                <div class="who">{html.escape(result['name'])} studied {result['hours']:g} hours</div>
                <div class="prob">{result['prob']:.0%} chance of passing</div></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="empty">No prediction yet. Fill in the form and press '
                        '<b>Predict result</b>.</div>', unsafe_allow_html=True)

        st.markdown('<h3 class="panel">How it works</h3>', unsafe_allow_html=True)
        st.markdown(
            f"""<ol class="how">
            <li>The app loads a logistic regression model saved earlier as <code>{MODEL_FILE}</code>. It is not retrained.</li>
            <li>The model turns the study hours into a probability of passing.</li>
            <li>Above 50% the class is PASS. At or below 50% it is FAIL.</li>
            </ol>""",
            unsafe_allow_html=True,
        )

with right:
    with st.container(border=True, key="p3"):
        st.markdown('<h3 class="panel">Study hours vs predicted class</h3>',
                    unsafe_allow_html=True)
        grid = np.linspace(0, 10, 101)
        curve = pd.DataFrame({
            "hours": grid,
            "prob": model.predict_proba(grid.reshape(-1, 1))[:, 1],
        })

        axis = dict(labelColor=MUTED, titleColor=MUTED, gridColor="#332E6E",
                    domainColor="#3A3478", tickColor="#3A3478")
        base = alt.Chart(curve).encode(
            x=alt.X("hours:Q", title="Study hours", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(**axis)),
            y=alt.Y("prob:Q", title="Probability of passing", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", **axis)),
        )
        layers = [
            base.mark_area(color=GOLD, opacity=0.07),
            base.mark_line(color=GOLD, strokeWidth=3),
            alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
                color=MUTED, strokeDash=[5, 5]).encode(y="y:Q"),
            alt.Chart(pd.DataFrame({"x": [7.6], "y": [0.54], "t": ["PASS above 50%"]})).mark_text(
                color=MINT, align="left", fontSize=12).encode(x="x:Q", y="y:Q", text="t:N"),
            alt.Chart(pd.DataFrame({"x": [0.2], "y": [0.46], "t": ["FAIL below 50%"]})).mark_text(
                color=CORAL, align="left", fontSize=12).encode(x="x:Q", y="y:Q", text="t:N"),
        ]
        if result:
            color = MINT if result["pred"] == 1 else CORAL
            pt = pd.DataFrame({"hours": [result["hours"]], "prob": [result["prob"]],
                               "student": [result["name"]]})
            layers.append(alt.Chart(pt).mark_rule(color=color, strokeDash=[2, 4]).encode(x="hours:Q"))
            layers.append(alt.Chart(pt).mark_point(
                filled=True, size=260, color=color, stroke=CHALK, strokeWidth=2).encode(
                x="hours:Q", y="prob:Q",
                tooltip=["student:N", alt.Tooltip("hours:Q", format=".1f"),
                         alt.Tooltip("prob:Q", format=".0%")]))

        chart = (alt.layer(*layers).properties(height=400)
                 .configure(background="transparent").configure_view(strokeWidth=0))
        st.altair_chart(chart, width="stretch")
        st.markdown('<div class="note">The gold curve is the model\'s probability of passing. '
                    'The dot marks the student you predicted.</div>', unsafe_allow_html=True)
