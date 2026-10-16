"""Polished Streamlit UI for the educational student pass prediction model."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"
st.set_page_config(page_title="Student Insight | Pass Predictor", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#182230;--muted:#667085;--line:#e7ebf2;--blue:#3559e0}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;font-size:16px}
*:focus-visible{outline:3px solid #8ba4ff!important;outline-offset:2px!important}
.stMarkdown p,.stMarkdown li{line-height:1.65}
.stApp{background:#f5f7fb;color:var(--ink)}
[data-testid="stHeader"]{background:rgba(245,247,251,.92)}
[data-testid="stSidebar"]{background:#111c35;border-right:0}
[data-testid="stSidebar"] *{color:#eef2ff!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#b8c4e2!important}
.brand{font-family:Manrope,sans-serif;font-size:1.2rem;font-weight:800;letter-spacing:-.04em;color:#fff}
.brand-mark{display:inline-flex;width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#6d83ff,#3a55df);align-items:center;justify-content:center;margin-right:10px}
.side-label{color:#9eadd0!important;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;margin-top:24px}
.hero{padding:28px 30px;border-radius:22px;background:linear-gradient(120deg,#18294f 0%,#263f85 60%,#3b59cf 100%);color:white;margin:4px 0 22px;box-shadow:0 14px 35px rgba(30,53,115,.15)}
.hero-kicker{color:#bdcaff;text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;font-weight:700}
.hero h1{color:#fff;font-family:Manrope,sans-serif;font-size:clamp(1.8rem,3vw,2.5rem);letter-spacing:-.05em;line-height:1.15;margin:.55rem 0 .7rem}
.hero p{color:#e1e8ff;font-size:1rem;margin:0;max-width:720px}
.section-title{font-family:Manrope,sans-serif;font-size:1.2rem;font-weight:800;letter-spacing:-.035em;margin:4px 0}
.section-sub{color:#667085;font-size:.94rem;line-height:1.55;margin-bottom:16px}
.helper-pill{display:inline-block;background:#eef2ff;color:#3047a8;border-radius:999px;padding:5px 10px;font-size:.78rem;font-weight:700;margin:4px 4px 4px 0}
.stCaption{line-height:1.5}
div[data-testid="stVerticalBlockBorderWrapper"]{background:white;border:1px solid var(--line);border-radius:18px;box-shadow:0 4px 16px rgba(23,36,68,.035)}
div[data-testid="stMetric"]{background:white;border:1px solid var(--line);padding:18px 20px;border-radius:16px}
div[data-testid="stMetricLabel"]{color:#667085;font-weight:600}
div[data-testid="stMetricValue"]{font-family:Manrope,sans-serif;color:#182230;font-weight:800}
.result-pass{padding:20px 22px;border-radius:16px;background:#ecfdf3;border:1px solid #b7ebcb}
.result-fail{padding:20px 22px;border-radius:16px;background:#fff1f0;border:1px solid #ffc9c4}
.result-label{font-size:.75rem;text-transform:uppercase;letter-spacing:.12em;font-weight:800;color:#667085}
.result-value{font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;margin:.2rem 0}
.result-note,.small-note{color:#526071;font-size:.88rem}
.stButton>button,.stFormSubmitButton>button{border-radius:11px;min-height:46px;font-weight:700;border:0;background:linear-gradient(100deg,#3559e0,#4a6cf0);color:white;box-shadow:0 5px 12px rgba(53,89,224,.18)}
.stButton>button:hover,.stFormSubmitButton>button:hover{border:0;background:#2948c8;color:white}
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div{border-radius:10px}
hr{border-color:#e7ebf2}footer{visibility:hidden}
@media(max-width:700px){.hero{padding:22px}}
</style>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("The trained model file is missing. Run `python train_model.py` in this project folder.")
    st.stop()
try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">🎓</span>Student Insight</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#b8c4e2;font-size:.82rem;margin-top:8px;">Academic performance toolkit</p>', unsafe_allow_html=True)
    st.markdown('<p class="side-label">WORKSPACE</p>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Overview", "Prediction", "Model insights"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p class="side-label">ABOUT THIS MODEL</p>', unsafe_allow_html=True)
    st.markdown("**Demo model**\n\nUses average daily study hours as its only input.")
    st.caption("Educational demo only. Predictions should never be used to grade or rank students.")
    if st.button("Clear latest result", use_container_width=True, help="Remove the saved result from this session."):
        st.session_state.pop("last_result", None)
        st.rerun()
    st.markdown("---")
    st.caption("Student Insight · Prototype")

if page == "Overview":
    st.markdown('''<div class="hero"><div class="hero-kicker">Student success • Predictive learning</div><h1>Make learning progress<br>easier to understand.</h1><p>Explore an educational machine-learning demo that estimates a pass/fail outcome from average daily study time.</p></div>''', unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("Model input", "Study hours", "0–24 hours/day")
    b.metric("Prediction classes", "2", "Pass or fail")
    c.metric("Model type", "Logistic regression", "Scaled pipeline")
    left,right=st.columns([1.15,.85],gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title">Get started</div><div class="section-sub">Run a sample prediction in a few steps.</div>',unsafe_allow_html=True)
            st.markdown("1. Open **Prediction** from the sidebar.\n2. Enter a student name and daily study hours.\n3. Select **Generate prediction** to view the model output.")
    with right:
        with st.container(border=True):
            st.markdown('<div class="section-title">Important context</div>',unsafe_allow_html=True)
            st.write("This model is trained on a small, illustrative dataset. Its output is a demonstration—not a guarantee of academic performance.")
            st.warning("Attendance is not used by the current model.",icon="ℹ️")
elif page == "Prediction":
    st.markdown('''<div class="hero"><div class="hero-kicker">Prediction workspace</div><h1>Student outcome estimator</h1><p>Enter the details below to generate a model-based estimate. Only study hours affect this prediction.</p></div>''',unsafe_allow_html=True)
    form_col,result_col=st.columns([1,1],gap="large")
    with form_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">Student details</div><div class="section-sub">Complete the required fields.</div>',unsafe_allow_html=True)
            with st.form("prediction_form",clear_on_submit=False):
                student_name=st.text_input("Student name *",placeholder="e.g. Alex Santos",help="Used only to label this result.")
                study_hours=st.slider("Average study hours per day",min_value=0.0,max_value=24.0,value=4.0,step=0.5,help="Estimate the average time spent studying each day. Use 0 if the student does not study daily.")
                st.markdown(f"**Selected study time:** `{study_hours:.1f} hours/day`")
                submitted=st.form_submit_button("✦  Generate prediction",use_container_width=True)
    with result_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">Prediction result</div><div class="section-sub">Your latest result appears here.</div>',unsafe_allow_html=True)
            if submitted:
                if not student_name.strip():
                    st.warning("Please enter the student's name before generating a prediction.")
                else:
                    prediction=int(model.predict([[float(study_hours)]])[0])
                    result="Pass" if prediction==1 else "Fail"
                    confidence=float(max(model.predict_proba([[float(study_hours)]])[0]))*100 if hasattr(model,"predict_proba") else None
                    css="result-pass" if result=="Pass" else "result-fail"
                    symbol="✓" if result=="Pass" else "!"
                    st.markdown(f'<div class="{css}"><div class="result-label">Estimated outcome</div><div class="result-value">{symbol} {result}</div><div class="result-note">Student: {student_name.strip()}</div></div>',unsafe_allow_html=True)
                    st.write("")
                    st.metric("Average study time",f"{study_hours:.1f} hrs/day")
                    if confidence is not None:
                        st.write("Model score (not a measure of real-world accuracy)")
                        st.progress(int(round(confidence)),text=f"{confidence:.1f}% model score")
                        st.caption("This score reflects the model's output, not a guarantee that the student will pass or fail.")
                    st.session_state["last_result"]={"student":student_name.strip(),"hours":float(study_hours),"result":result,"confidence":confidence}
            elif st.session_state.get("last_result"):
                last=st.session_state["last_result"]
                css="result-pass" if last["result"]=="Pass" else "result-fail"
                st.markdown(f'<div class="{css}"><div class="result-label">Latest estimated outcome</div><div class="result-value">{last["result"]}</div><div class="result-note">Student: {last["student"]} · {last["hours"]:.1f} study hours/day</div></div>',unsafe_allow_html=True)
            else:
                st.info("Complete the student details and generate a prediction to see the result here.")
            st.markdown("---")
            st.markdown('<p class="small-note">This is a classroom demo trained on a tiny illustrative dataset. Do not use it to grade, rank, or make decisions about students.</p>',unsafe_allow_html=True)
else:
    st.markdown('''<div class="hero"><div class="hero-kicker">Transparency & limitations</div><h1>Understand the model</h1><p>Review how the current demo behaves across a range of study-hour values.</p></div>''',unsafe_allow_html=True)
    hours=[float(x) for x in range(0,25,2)]
    chart_data=pd.DataFrame({"Study hours per day":hours})
    chart_data["Predicted outcome"]=["Pass" if int(model.predict([[h]])[0])==1 else "Fail" for h in hours]
    if hasattr(model,"predict_proba"):
        chart_data["Pass probability (%)"]=[round(float(model.predict_proba([[h]])[0][1])*100,1) for h in hours]
    left,right=st.columns([1.15,.85],gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title">Predicted outcome by study time</div><div class="section-sub">Each row is evaluated by the saved model.</div>',unsafe_allow_html=True)
            st.dataframe(chart_data,use_container_width=True,hide_index=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="section-title">How to read this</div>',unsafe_allow_html=True)
            st.markdown("- **Input:** average study hours per day.\n- **Output:** predicted class, Pass or Fail.\n- **Algorithm:** logistic regression with feature scaling.\n- **Limit:** attendance, grades, and other factors are not model inputs.")
            if "Pass probability (%)" in chart_data:
                st.markdown("**Pass probability trend**")
                st.line_chart(chart_data.set_index("Study hours per day")["Pass probability (%)"])
    st.info("A model probability is not the same as a student's true chance of passing. The training data is small and illustrative; validate on representative data before drawing conclusions.")
