from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Performance | Rica Alba Prequencia",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080d17; color: #e8eef9; }
[data-testid="stHeader"] { background: rgba(8,13,23,0.95); }
[data-testid="stSidebar"] { background: #0d1524; border-right: 1px solid #1d2a40; }
[data-testid="stSidebar"] * { color: #dce7f8; }
.block-container { max-width: 1380px; padding-top: 2rem; padding-bottom: 3rem; }
.hero { padding: 30px 32px; border-radius: 22px; background: linear-gradient(120deg,#102442,#101a2b 62%,#102d45); border: 1px solid #203c5e; margin-bottom: 24px; }
.eyebrow { color: #63b3ff; text-transform: uppercase; font-size: 11px; font-weight: 800; letter-spacing: 2px; }
.hero h1 { color: #f4f8ff; font-size: clamp(28px,3vw,40px); line-height: 1.15; margin: 10px 0; font-weight: 800; }
.hero p { color: #a9bad2; font-size: 14px; margin-bottom: 0; }
.section-title { color: #f3f7ff; font-size: 18px; font-weight: 700; margin: 8px 0 14px; }
.panel { background: #0e1727; border: 1px solid #1e2c42; border-radius: 18px; padding: 22px; height: 100%; }
.metric-label { color: #8fa4c1; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
.metric-value { color: #f5f8ff; font-size: 27px; font-weight: 800; margin-top: 7px; }
.muted { color: #91a4bf; font-size: 13px; line-height: 1.6; }
div.stButton > button { background: #2385f4; color: white; border: 0; border-radius: 10px; padding: 0.68rem 1rem; font-weight: 700; width: 100%; }
div.stButton > button:hover { background: #3d97ff; color: white; border: 0; }
.stTextInput input, .stNumberInput input { background: #0a1220; color: #f2f6ff; border: 1px solid #293b55; border-radius: 10px; }
div[data-baseweb="select"] > div { background: #0a1220; border-color: #293b55; border-radius: 10px; }
hr { border-color: #1c2b40; }
[data-testid="stMetric"] { background: #0e1727; border: 1px solid #1e2c42; border-radius: 14px; padding: 16px; }
[data-testid="stMetricLabel"] { color: #91a4bf; }
[data-testid="stMetricValue"] { color: #f5f8ff; }
div[data-testid="stAlert"] { border-radius: 12px; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 Student Insight")
    st.caption("PERFORMANCE PREDICTION SYSTEM")
    st.divider()
    st.markdown("**Student / Developer**")
    st.write("Rica Alba Prequencia")
    st.divider()
    st.markdown("**Navigation**")
    st.markdown("🔹 Prediction Dashboard")
    st.markdown("🔹 Model Overview")
    st.divider()
    st.caption("Machine Learning • Flask • Streamlit")
    st.caption("Academic demonstration only")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Student analytics · Machine learning</div>
  <h1>Student Performance Dashboard</h1>
  <p>Estimate a student's result from study hours using a saved machine-learning model.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("The saved model was not found. In the project terminal, run: `python train_model.py`")
    st.stop()

@st.cache_resource
def get_model(path):
    return joblib.load(path)

model = get_model(str(MODEL_PATH))

st.markdown('<div class="section-title">Prediction workspace</div>', unsafe_allow_html=True)
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Student details")
    st.markdown("<p class='muted'>Enter the student's details, then generate a prediction.</p>", unsafe_allow_html=True)
    with st.form("prediction_form"):
        student_name = st.text_input("Student name", value="Rica Alba Prequencia", placeholder="Enter full name")
        study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        attendance = st.number_input("Attendance percentage", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
        submitted = st.form_submit_button("Generate prediction", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Prediction result")
    if submitted:
        if not student_name.strip():
            st.warning("Please enter the student's name.")
        else:
            prediction = int(model.predict([[study_hours]])[0])
            probability = float(model.predict_proba([[study_hours]])[0][1])
            result = "Pass" if prediction == 1 else "Needs Improvement"
            st.session_state["last_prediction"] = {
                "name": student_name.strip(), "hours": study_hours,
                "attendance": attendance, "result": result, "probability": probability
            }
    last = st.session_state.get("last_prediction")
    if last:
        is_pass = last["result"] == "Pass"
        color = "#3bd6a0" if is_pass else "#ffbf69"
        st.markdown(f'<div style="color:#91a4bf;font-size:12px;text-transform:uppercase;letter-spacing:1px">Result for</div><div style="font-size:20px;font-weight:700;margin:5px 0 16px">{last["name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#0a1220;border:1px solid #263a55;border-radius:14px;padding:20px;margin-bottom:16px"><div style="color:{color};font-size:29px;font-weight:800">{last["result"]}</div><div class="muted">Model-estimated pass probability: {last["probability"]:.1%}</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Study hours/day", f'{last["hours"]:g} h')
        c2.metric("Attendance", f'{last["attendance"]:g}%')
        if is_pass:
            st.success("The model classifies this study-hours input as Pass.")
        else:
            st.warning("The model classifies this study-hours input as Needs Improvement.")
    else:
        st.info("Your prediction will appear here after you submit the form.")
        st.markdown('<p class="muted">The result is generated by loading the saved model, not by retraining it on every prediction.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown('<div class="section-title">Study-hours comparison</div>', unsafe_allow_html=True)
chart_col, explanation_col = st.columns([1.2, 1], gap="large")
with chart_col:
    hours_range = list(range(0, 11))
    labels = ["Pass" if int(model.predict([[h]])[0]) == 1 else "Needs Improvement" for h in hours_range]
    chart_df = pd.DataFrame({"Study hours per day": hours_range, "Predicted class": [1 if x == "Pass" else 0 for x in labels]})
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0e1727")
    ax.set_facecolor("#0e1727")
    ax.bar(chart_df["Study hours per day"], chart_df["Predicted class"])
    ax.set_yticks([0, 1], ["Needs Improvement", "Pass"])
    ax.set_xticks(hours_range)
    ax.set_xlabel("Study hours per day", color="#a9bad2")
    ax.tick_params(colors="#a9bad2")
    for spine in ax.spines.values():
        spine.set_color("#293b55")
    ax.grid(axis="y", alpha=0.15)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with explanation_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### How it works")
    st.markdown("""
1. The trained model is saved as `student_pass_model.joblib`.
2. Streamlit loads the saved model with `joblib.load()`.
3. The model receives the study-hours value and predicts a class.
4. The dashboard displays the result and comparison chart.
""")
    st.markdown('<p class="muted"><b>Important:</b> Attendance is validated and shown in the interface, but it is not a model feature in this exercise. The training data is illustrative and this demo is not an actual academic assessment.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center;color:#657a97;font-size:12px">STUDENT PERFORMANCE PREDICTION · Rica Alba Prequencia</p>', unsafe_allow_html=True)
