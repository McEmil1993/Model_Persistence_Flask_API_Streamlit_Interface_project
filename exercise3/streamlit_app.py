"""Exercise 3 - Streamlit interface for the study-hours model (report-card design)."""
import datetime
import html

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Report Card Predictor", page_icon="📝", layout="wide")

INK = "#1F2937"
PAPER = "#FAF6EE"
CARD = "#FFFFFF"
LINE = "#E5DCCB"
MUTED = "#6B7280"
PASS = "#2F855A"
FAIL = "#C2410C"

st.markdown(
    f"""
    <style>
      .stApp {{ background: {PAPER}; color: {INK}; }}
      .block-container {{ padding-top: 2.2rem; max-width: 1150px; }}
      h1, h2, h3, .serif {{ font-family: Georgia, 'Times New Roman', serif !important; color: {INK}; }}
      label, .stMarkdown, p, li {{ color: {INK}; }}
      #MainMenu, footer, header {{ visibility: hidden; }}

      .masthead {{ border-bottom: 3px double {INK}; padding-bottom: 14px; margin-bottom: 26px; }}
      .masthead h1 {{ font-size: 2.3rem; margin: 0; padding: 0; letter-spacing: .5px; }}
      .masthead p {{ color: {MUTED}; margin: .3rem 0 0 0; font-style: italic; }}

      div[data-testid="stForm"] {{
        background: {CARD}; border: 1px solid {LINE}; border-radius: 6px;
        padding: 22px 22px 14px 22px; box-shadow: 4px 4px 0 {LINE};
      }}
      div[data-testid="stFormSubmitButton"] > button {{
        width: 100%; border-radius: 4px; font-weight: 700; letter-spacing: .5px;
        background: {INK}; color: #fff; border: none; padding: .6rem 0;
      }}
      div[data-testid="stFormSubmitButton"] > button:hover {{ background: #000; color: #fff; }}
      div[data-testid="stFormSubmitButton"] button,
      div[data-testid="stFormSubmitButton"] button * {{ color: #FFFFFF !important; }}

      .card {{
        position: relative; background: {CARD}; border: 1px solid {LINE}; border-radius: 6px;
        padding: 22px 26px; box-shadow: 4px 4px 0 {LINE}; margin-bottom: 18px;
      }}
      .card-head {{
        display: flex; justify-content: space-between; font-size: .75rem; letter-spacing: 2px;
        color: {MUTED}; text-transform: uppercase; border-bottom: 1px solid {LINE};
        padding-bottom: 8px; margin-bottom: 14px;
      }}
      .student {{ font-size: 1.8rem; margin: 0 0 14px 0 !important; padding: 0 !important; }}
      .row {{ display: flex; gap: 28px; align-items: center; flex-wrap: wrap; }}
      .gauge {{
        width: 138px; height: 138px; border-radius: 50%; flex: none;
        background: conic-gradient(var(--c) calc(var(--p) * 1%), {LINE} 0);
        display: grid; place-items: center;
      }}
      .gauge-in {{
        width: 104px; height: 104px; border-radius: 50%; background: {CARD};
        display: flex; flex-direction: column; align-items: center; justify-content: center;
      }}
      .gauge-in b {{ font-size: 1.8rem; font-family: Georgia, serif; color: {INK}; }}
      .gauge-in small {{ color: {MUTED}; font-size: .7rem; }}
      .facts div {{ margin-bottom: 8px; }}
      .facts small {{ display: block; color: {MUTED}; font-size: .72rem; text-transform: uppercase; letter-spacing: 1px; }}
      .facts b {{ font-size: 1.05rem; color: {INK}; }}
      .stamp {{
        position: absolute; top: 62px; right: 26px; transform: rotate(-9deg);
        border: 4px solid currentColor; border-radius: 6px; padding: 2px 16px;
        font-family: Georgia, serif; font-weight: 800; font-size: 1.9rem; letter-spacing: 3px; opacity: .88;
      }}
      .placeholder {{ text-align: center; color: {MUTED}; padding: 56px 10px; font-style: italic; }}
      .steps {{ margin: 0; padding-left: 1.2rem; }}
      .steps li {{ margin-bottom: 6px; }}
    </style>
    <div class="masthead">
      <h1>📝 Student Report Card Predictor</h1>
      <p>Tell us who is studying and for how long. The model tells you how likely they are to pass.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

if getattr(model, "n_features_in_", 1) != 2:
    st.error("This app needs the two-feature model (study hours + attendance). "
             "Run `python train_model.py` in the exercise2 folder and copy "
             "`student_pass_model.joblib` into this folder, then refresh.")
    st.stop()

left, right = st.columns([2, 3], gap="large")

# ---------------- Left: inputs ----------------
with left:
    with st.form("prediction_form"):
        st.markdown("### Student details")
        name = st.text_input("Student name", placeholder="e.g. Maria Santos")
        hours = st.slider("Study hours per day", min_value=0.0, max_value=12.0,
                          value=3.0, step=0.5)
        attendance = st.slider("Attendance percentage (%)", min_value=0, max_value=100,
                               value=80, step=1)
        submitted = st.form_submit_button("PREDICT RESULT")

# ---------------- Right: report card ----------------
with right:
    if not submitted:
        st.markdown(
            '<div class="card"><div class="card-head"><span>Report card</span><span>Pending</span></div>'
            '<div class="placeholder">Fill in the form and press <b>Predict result</b>.<br>'
            "The student's report card will appear here.</div></div>",
            unsafe_allow_html=True)
        st.stop()

    if not name.strip():
        st.warning("Please enter the student's name.")
        st.stop()

    features = pd.DataFrame({"study_hours": [hours], "attendance_percentage": [attendance]})
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])
    label = "Pass" if pred == 1 else "Fail"
    color = PASS if pred == 1 else FAIL
    safe_name = html.escape(name.strip())
    today = datetime.date.today().strftime("%B %d, %Y")

    st.markdown(
        f"""
        <div class="card">
          <div class="card-head"><span>Report card</span><span>{today}</span></div>
          <h2 class="student">{safe_name}</h2>
          <div class="stamp" style="color:{color}">{label.upper()}</div>
          <div class="row">
            <div class="gauge" style="--p:{prob * 100:.1f}; --c:{color}">
              <div class="gauge-in"><b>{prob:.0%}</b><small>pass chance</small></div>
            </div>
            <div class="facts">
              <div><small>Study hours</small><b>{hours:g} h per day</b></div>
              <div><small>Attendance</small><b>{attendance}%</b></div>
              <div><small>Predicted class</small><b>{pred} &middot; {label}</b></div>
              <div><small>Decision rule</small><b>50% or more = Pass</b></div>
            </div>
          </div>
        </div>
        <div class="card">
          <div class="card-head"><span>How this was generated</span></div>
          <ol class="steps">
            <li>A logistic regression model, trained earlier and saved with joblib, was loaded. It was <b>not</b> retrained.</li>
            <li>It took <b>{hours:g}</b> study hours and <b>{attendance}%</b> attendance as its two inputs.</li>
            <li>It combined them into a pass probability of <b>{prob:.0%}</b>.</li>
            <li>Because that is {"at least" if pred else "below"} 50%, the predicted class is <b>{pred} ({label})</b>.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Bonus chart ----------------
    st.markdown("### Study hours vs predicted class")
    grid = np.arange(0, 10.01, 0.25)
    probs = model.predict_proba(pd.DataFrame({"study_hours": grid, "attendance_percentage": attendance}))[:, 1]
    df = pd.DataFrame({
        "hours": grid,
        "probability": probs,
        "class": np.where(probs >= 0.5, "Pass (1)", "Fail (0)"),
    })
    shown = min(hours, 10)

    bars = alt.Chart(df).mark_bar(size=9).encode(
        x=alt.X("hours:Q", title="Study hours per day", scale=alt.Scale(domain=[0, 10.25])),
        y=alt.Y("probability:Q", title="Pass probability",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("class:N", title="Predicted class",
                        scale=alt.Scale(domain=["Fail (0)", "Pass (1)"], range=[FAIL, PASS])),
        tooltip=[alt.Tooltip("hours:Q", title="Study hours"),
                 alt.Tooltip("probability:Q", title="Pass probability", format=".0%"),
                 alt.Tooltip("class:N", title="Predicted class")],
    )
    rule = alt.Chart(pd.DataFrame({"hours": [shown]})).mark_rule(
        color=INK, strokeWidth=2, strokeDash=[4, 3]).encode(x="hours:Q")
    tag = alt.Chart(pd.DataFrame({"hours": [shown], "text": [f"{name.strip()} ({hours:g} h)"]})).mark_text(
        align="left", dx=6, baseline="top", color=INK, fontWeight="bold").encode(
        x="hours:Q", y=alt.value(4), text="text:N")

    chart = (bars + rule + tag).properties(height=300, background=CARD).configure_view(
        stroke=LINE).configure_axis(labelColor=INK, titleColor=INK, gridColor=LINE)
    st.altair_chart(chart, width="stretch")
    st.caption(f"Each bar is the model's pass probability at that many study hours (with attendance held at "
               f"{attendance}%), coloured by the class it predicts. The dashed line marks the student entered above."
               + (" Hours above 10 are shown at the edge of the chart." if hours > 10 else ""))