"""Student pass/fail studio -- dashboard layout with sidebar controls."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).with_name("student_pass_model.joblib")
clf = joblib.load(MODEL_FILE)

st.set_page_config(
    page_title="Pass / Fail Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #0e7490 55%, #10b981 100%);
        border-radius: 18px; padding: 24px 26px;
        box-shadow: 0 10px 28px rgba(14,116,144,.30);
    }
    .hero h1, .hero p, .hero span { color: #ffffff !important; }
    .hero h1 { margin: 0; font-size: 1.9rem; }
    .hero p { margin: 6px 0 0 0; opacity: .95; }
    .pill {
        display: inline-block; background: rgba(255,255,255,.15);
        border: 1px solid rgba(255,255,255,.4);
        padding: 3px 12px; border-radius: 999px;
        font-size: .78rem; margin-right: 6px; margin-top: 10px;
    }
    section[data-testid="stSidebar"] { background: transparent; }
    div.stButton > button {
        background: #059669; border-radius: 10px;
        padding: .6rem 1rem; font-weight: 700;
    }
    div.stButton > button p,
    div.stButton > button:hover p,
    div.stButton > button:focus p { color: #ffffff !important; }
    .result-card {
        border-radius: 14px; padding: 18px 20px;
        border: 1px solid rgba(128,128,128,.35);
        border-left: 8px solid #64748b;
    }
    .result-pass { border-left-color: #10b981; }
    .result-fail { border-left-color: #ef4444; }
    .result-verdict-pass { color: #059669 !important; font-size: 2rem; font-weight: 800; }
    .result-verdict-fail { color: #dc2626 !important; font-size: 2rem; font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar controls (NEW position: inputs moved out of main columns) ----------
with st.sidebar:
    st.header("🎓 Controls")
    st.caption("All inputs live here now — results show on the right.")
    full_name = st.text_input(
        "Student name", placeholder="e.g. Sam Morgan",
        help="Just a label — it does not affect the model.",
    )
    hours = st.slider(
        "Study hours per week", 0.0, 12.0, 4.5, 0.5,
        help="Main driver of the prediction. Try 1.5 vs 9.0.",
    )
    attendance = st.slider(
        "Attendance (%)", 0, 100, 80, 1,
        help="Must stay within 0–100. Fixed for the charts below.",
    )
    run = st.button("✨ Run prediction", type="primary", use_container_width=True)
    st.divider()
    if hours < 3:
        st.caption("📚 Under 3 hrs/week usually predicts FAIL.")
    elif hours < 6:
        st.caption("📈 3–6 hrs is the borderline zone.")
    else:
        st.caption("🚀 6+ hrs usually predicts PASS.")

# ---------- Main dashboard ----------
st.markdown(
    """
    <div class="hero">
      <h1>🎓 Pass / Fail Dashboard</h1>
      <p>Sidebar controls → instant prediction → side-by-side charts. Same saved model, new workspace.</p>
      <span class="pill">Sidebar inputs</span>
      <span class="pill">Top result bar</span>
      <span class="pill">Dual charts</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if run:
    chance = float(clf.predict_proba([[hours, attendance]])[0][1])
    passed = chance >= 0.5
    verdict = "PASS" if passed else "FAIL"
    who = full_name.strip() or "This student"
    card = "result-pass" if passed else "result-fail"
    vclass = "result-verdict-pass" if passed else "result-verdict-fail"
    icon = "✅" if passed else "❌"

    st.markdown(
        f"<div class='result-card {card}'>"
        f"<div>{icon} <b>{who}</b> — P(pass) = {chance:.2f}</div>"
        f"<div class='{vclass}'>{verdict}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of passing", f"{chance:.2%}")
    m2.metric("Study hours", f"{hours:.1f} hrs")
    m3.metric("Attendance", f"{attendance}%")
    st.progress(min(max(chance, 0.0), 1.0))

    if passed:
        st.success(f"{who} is predicted to {verdict}. Keep it up!")
        st.balloons()
    else:
        st.error(f"{who} is predicted to {verdict}. More hours + attendance help.")
else:
    with st.container(border=True):
        st.info("👈 No prediction yet. Use the **sidebar** and click **Run prediction**.")

st.write("")

# ---------- Charts: side-by-side (NEW position: was stacked tabs, now dual columns) ----------
hour_grid = [round(x * 0.5, 1) for x in range(0, 25)]
probs = [float(clf.predict_proba([[h, attendance]])[0][1]) for h in hour_grid]
classes = [int(p >= 0.5) for p in probs]
trend = pd.DataFrame(
    {"chance_pass": probs, "pass_class": classes}, index=hour_grid
)
trend.index.name = "hours"

st.subheader(f"📊 Hours vs outcome (attendance = {attendance}%)")
chart_left, chart_right = st.columns(2, gap="large")
with chart_left:
    with st.container(border=True):
        st.markdown("**Probability curve**")
        st.area_chart(trend[["chance_pass"]])
with chart_right:
    with st.container(border=True):
        st.markdown("**Predicted class (0 = FAIL, 1 = PASS)**")
        st.bar_chart(trend[["pass_class"]])

# ---------- Explanation + data: bottom split (NEW position) ----------
bottom_left, bottom_right = st.columns([1.2, 1], gap="large")
with bottom_left:
    with st.expander("💡 How was this generated?", expanded=True):
        st.write(
            "Your hours and attendance are standardised with the scaler "
            "stored in `student_pass_model.joblib`, fed to logistic "
            "regression for P(pass), then cut at 0.50 for PASS/FAIL. "
            "The file is loaded once — no retraining happens here."
        )
with bottom_right:
    with st.expander("🗂 Sweep data"):
        st.dataframe(trend, use_container_width=True)

st.divider()
st.caption("Model: `student_pass_model.joblib` · Threshold 0.50 · No retraining at predict time.")
