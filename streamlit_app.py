"""Exercise 3 - Streamlit interface for the student pass predictor (redesigned)."""
import html

import altair as alt
import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "student_pass_model.joblib"

# Palette
INK = "#14213D"
PAPER = "#F7F8FC"
HIGHLIGHT = "#FFD84D"
PASS = "#1B8A5A"
FAIL = "#D6404A"
MUTED = "#5B6784"

st.set_page_config(page_title="Study hours predictor", page_icon="📘", layout="centered")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Source+Sans+3:wght@400;600&display=swap');

:root {{ --ink:{INK}; --paper:{PAPER}; --hl:{HIGHLIGHT}; --pass:{PASS}; --fail:{FAIL}; --muted:{MUTED}; }}

.stApp {{
    background-color: var(--paper);
    background-image:
        linear-gradient(rgba(20,33,61,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(20,33,61,.045) 1px, transparent 1px);
    background-size: 28px 28px;
    color: var(--ink);
    font-family: 'Source Sans 3', sans-serif;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 720px; padding-top: 2.5rem; padding-bottom: 4rem; }}

h1, h2, h3, .slip-title {{ font-family: 'Bricolage Grotesque', sans-serif !important; color: var(--ink) !important; letter-spacing: -0.02em; }}

.hero h1 {{ font-size: clamp(2.2rem, 6vw, 3.4rem); font-weight: 800; line-height: 1.02; margin: 0 0 .6rem 0; padding: 0; }}
.hero mark {{ background: linear-gradient(transparent 55%, var(--hl) 55%); color: inherit; padding: 0 .1em; }}
.hero p {{ font-size: 1.1rem; color: var(--muted); max-width: 46ch; margin: 0 0 1.6rem 0; line-height: 1.5; }}

/* form */
label, .stTextInput label p, .stNumberInput label p {{ color: var(--ink) !important; font-weight: 600 !important; }}
.stTextInput input, .stNumberInput input {{
    background: #fff !important; color: var(--ink) !important;
    border: 2px solid var(--ink) !important; border-radius: 10px !important;
    font-size: 1.05rem !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{ box-shadow: 4px 4px 0 var(--hl) !important; }}
div[data-baseweb="input"], div[data-baseweb="base-input"] {{ background: transparent !important; border: none !important; }}
.stNumberInput button {{ color: var(--ink) !important; background: #fff !important; border: 2px solid var(--ink) !important; }}

.stButton > button {{
    background: var(--ink); color: #fff; border: 2px solid var(--ink);
    border-radius: 10px; padding: .65rem 1.6rem; font-weight: 600; font-size: 1.05rem;
    box-shadow: 4px 4px 0 var(--hl); transition: transform .12s ease, box-shadow .12s ease;
}}
.stButton > button:hover {{ background: var(--ink); color: #fff; border-color: var(--ink); transform: translate(2px, 2px); box-shadow: 2px 2px 0 var(--hl); }}
.stButton > button:focus-visible {{ outline: 3px solid var(--hl); outline-offset: 2px; }}

/* result slip */
.slip {{
    background: #fff; border: 2px solid var(--ink); border-radius: 16px;
    padding: 1.6rem 1.8rem; margin: 1.6rem 0 1.2rem 0;
    box-shadow: 8px 8px 0 var(--ink);
    display: flex; gap: 1.6rem; align-items: center; flex-wrap: wrap;
}}
.ring {{
    --p: 0; --c: var(--pass);
    width: 132px; height: 132px; border-radius: 50%; flex: none;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), #E4E8F2 0);
    display: grid; place-items: center;
}}
.ring-inner {{
    width: 98px; height: 98px; border-radius: 50%; background: #fff;
    display: grid; place-items: center; text-align: center;
    font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.9rem; color: var(--ink);
    line-height: 1;
}}
.ring-inner small {{ display: block; font-family: 'Source Sans 3', sans-serif; font-weight: 400; font-size: .72rem; color: var(--muted); margin-top: .25rem; }}
.slip-body {{ flex: 1; min-width: 220px; }}
.slip-name {{ font-size: .95rem; color: var(--muted); margin: 0 0 .15rem 0; }}
.slip-title {{ font-size: 2rem; font-weight: 800; margin: 0 0 .4rem 0; line-height: 1.05; }}
.slip-title.pass {{ color: var(--pass) !important; }}
.slip-title.fail {{ color: var(--fail) !important; }}
.slip-text {{ margin: 0; color: var(--ink); line-height: 1.5; }}

/* explanation */
.explain {{ border-left: 6px solid var(--hl); background: #fff; padding: .9rem 1.2rem; border-radius: 0 10px 10px 0; line-height: 1.55; }}
.explain b {{ font-family: 'Bricolage Grotesque', sans-serif; }}

h3 {{ font-size: 1.35rem !important; font-weight: 700 !important; margin-top: 1.8rem !important; }}

@media (prefers-reduced-motion: reduce) {{ .stButton > button {{ transition: none; }} }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


model = load_model()

st.markdown(
    """
<div class="hero">
  <h1>Will they <mark>pass</mark>?</h1>
  <p>Enter a student's name and daily study hours. The model gives a pass or fail prediction and shows how sure it is.</p>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 2])
with col1:
    name = st.text_input("Student name", placeholder="Maria Santos")
with col2:
    hours = st.number_input(
        "Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5
    )

predict = st.button("Predict result")

if predict:
    if not name.strip():
        st.warning("Enter the student's name to see a result.")
    else:
        X = pd.DataFrame({"study_hours": [hours]})
        label = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
        shown = prob if label == 1 else 1 - prob  # confidence in the predicted class
        safe_name = html.escape(name.strip())
        cls = "pass" if label == 1 else "fail"
        verdict = "Likely to pass" if label == 1 else "Likely to fail"
        color = "var(--pass)" if label == 1 else "var(--fail)"

        st.markdown(
            f"""
<div class="slip">
  <div class="ring" style="--p:{shown*100:.0f}; --c:{color};">
    <div class="ring-inner"><div>{shown:.0%}<small>confidence</small></div></div>
  </div>
  <div class="slip-body">
    <p class="slip-name">{safe_name} &middot; {hours:g} study hours a day</p>
    <p class="slip-title {cls}">{verdict}</p>
    <p class="slip-text">The model puts the chance of passing at {prob:.0%}.</p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("### How this prediction was made")
        st.markdown(
            """
<div class="explain">
The app loads a logistic regression model that was trained earlier and saved with
<b>joblib</b>. It scales the study hours and turns them into a pass probability.
At 50% or higher the class is <b>Pass (1)</b>, otherwise <b>Fail (0)</b>.
Clicking the button does not retrain the model.
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("### Study hours and predicted class")
        grid = pd.DataFrame({"study_hours": [i / 4 for i in range(0, 49)]})
        grid["pass_probability"] = model.predict_proba(grid)[:, 1]
        point = pd.DataFrame({"study_hours": [hours], "pass_probability": [prob]})
        line_color = PASS if label == 1 else FAIL

        curve = (
            alt.Chart(grid)
            .mark_line(color=INK, strokeWidth=3)
            .encode(
                x=alt.X("study_hours:Q", title="Study hours per day"),
                y=alt.Y(
                    "pass_probability:Q",
                    title="Chance of passing",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
            )
        )
        threshold = (
            alt.Chart(pd.DataFrame({"y": [0.5]}))
            .mark_rule(color=MUTED, strokeDash=[6, 4])
            .encode(y="y:Q")
        )
        dot = (
            alt.Chart(point)
            .mark_point(
                filled=True, size=260, color=line_color, stroke=INK, strokeWidth=2
            )
            .encode(x="study_hours:Q", y="pass_probability:Q")
        )
        chart = (
            (curve + threshold + dot)
            .properties(height=300)
            .configure(background="#FFFFFF")
            .configure_view(stroke=INK, strokeWidth=2)
            .configure_axis(
                labelColor=INK,
                titleColor=INK,
                gridColor="#E4E8F2",
                labelFontSize=12,
                titleFontSize=13,
            )
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            "The dashed line is the 50% pass threshold. The dot marks "
            f"{safe_name}'s study hours."
        )