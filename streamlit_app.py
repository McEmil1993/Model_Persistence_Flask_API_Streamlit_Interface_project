
from pathlib import Path
import html
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --bg: #080d18;
    --surface: #101827;
    --surface-light: #151f31;
    --border: #25334a;
    --text: #edf4ff;
    --muted: #93a4bd;
    --blue: #3984ff;
    --cyan: #45d5f2;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(35, 89, 171, .17), transparent 35%),
        radial-gradient(ellipse at 90% 15%, rgba(24, 148, 184, .09), transparent 30%),
        var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(8, 13, 24, .85);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
}

p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: var(--muted);
}

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, #111d34 0%, #142d50 55%, #10445b 100%);
    border: 1px solid #2c4868;
    border-radius: 24px;
    padding: 2.5rem 2.6rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 18px 50px rgba(0, 0, 0, .20);
}

.hero::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -65px;
    top: -100px;
    border-radius: 50%;
    border: 1px solid rgba(69, 213, 242, .20);
    box-shadow: 0 0 0 35px rgba(69, 213, 242, .035),
                0 0 0 70px rgba(69, 213, 242, .025);
    pointer-events: none;
}

.eyebrow {
    color: var(--cyan);
    font-size: .73rem;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    letter-spacing: -.045em;
    margin: .65rem 0 .6rem;
    color: #ffffff !important;
}

.hero p {
    color: #bdcde3;
    font-size: 1rem;
    max-width: 650px;
    margin: 0;
    line-height: 1.7;
}

.section-heading {
    font-family: 'Manrope', sans-serif;
    color: var(--text);
    font-size: 1.15rem;
    font-weight: 800;
    margin-bottom: .25rem;
}

.section-description {
    color: var(--muted);
    font-size: .87rem;
    margin-bottom: 1.2rem;
}

.panel {
    background: linear-gradient(145deg, #111a2a, #0e1624);
    border: 1px solid var(--border);
    border-radius: 19px;
    padding: 1.5rem;
    box-shadow: 0 12px 30px rgba(0, 0, 0, .12);
    margin-bottom: 1.1rem;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 1.15rem;
    min-height: 105px;
}

.metric-label {
    color: var(--muted);
    font-size: .76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .07em;
}

.metric-value {
    color: #f2f7ff;
    font-family: 'Manrope', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    margin-top: .4rem;
}

.result-pass, .result-fail, .result-empty {
    border-radius: 16px;
    padding: 1.45rem;
    margin: .5rem 0 1.2rem;
}

.result-pass {
    background: linear-gradient(120deg, rgba(16, 185, 129, .13), rgba(16, 185, 129, .04));
    border: 1px solid rgba(52, 211, 153, .38);
}

.result-fail {
    background: linear-gradient(120deg, rgba(248, 113, 113, .13), rgba(248, 113, 113, .04));
    border: 1px solid rgba(248, 113, 113, .38);
}

.result-empty {
    background: var(--surface-light);
    border: 1px dashed #354761;
    padding: 2rem 1.3rem;
    text-align: center;
}

.result-label {
    color: var(--muted);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .14em;
}

.result-value {
    font-family: 'Manrope', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.3;
    margin: .25rem 0;
}

.result-pass .result-value {
    color: #52e0b0;
}

.result-fail .result-value {
    color: #ff8e8e;
}

.result-name {
    color: #c1cfe2;
    font-size: .9rem;
}

div[data-testid="stForm"] {
    border: none;
    padding: 0;
}

.stTextInput input {
    background: #0b1321 !important;
    color: #edf4ff !important;
    border: 1px solid #2b3b54 !important;
    border-radius: 10px !important;
}

.stTextInput input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 1px var(--blue) !important;
}

.stSlider [data-baseweb="slider"] {
    padding-top: .5rem;
}

.stSlider [data-testid="stTickBar"] {
    background: #34445c;
}

.stSlider [role="slider"] {
    background: var(--blue);
    border-color: var(--blue);
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(100deg, #2875f5, #248fd0);
    color: white;
    border: 1px solid #3984ff;
    border-radius: 11px;
    min-height: 3.1rem;
    font-weight: 700;
    transition: all .2s ease;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(100deg, #3984ff, #31a6e0);
    border-color: #65b8ff;
    color: white;
    box-shadow: 0 6px 20px rgba(57, 132, 255, .2);
}

.stProgress > div > div {
    background: linear-gradient(90deg, #3984ff, #45d5f2);
}

[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #202d41;
}

[data-testid="stSidebar"] > div:first-child {
    background: #0b1220;
}

[data-testid="stSidebar"] * {
    color: #dce7f7;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #899bb5;
}

[data-testid="stAlert"] {
    background: #142239;
    border: 1px solid #304663;
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background: #0c1523;
    border: 1px solid #25334a;
    border-radius: 12px;
    padding: 1rem;
}

[data-testid="stMetricLabel"] {
    color: #93a4bd;
}

[data-testid="stMetricValue"] {
    color: #edf4ff;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}

hr {
    border-color: #26364c;
}

.footer {
    color: #71829c;
    font-size: .78rem;
    text-align: center;
    padding: 1.5rem 0 .5rem;
}

@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        padding: 1.6rem;
        border-radius: 18px;
    }

    .panel {
        padding: 1.1rem;
    }

    .result-value {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(
        "The trained model is missing. Open a terminal in this project folder "
        "and run `python train_model.py` first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

# Header
st.markdown("""
<div class="hero">
    <div class="eyebrow">Student analytics / Machine learning</div>
    <h1>Student Pass Predictor</h1>
    <p>
        Analyze study habits and explore a machine-learning estimate of a
        student's academic outcome through a clean, interactive dashboard.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎓 Student Predictor")
    st.caption("ACADEMIC ANALYTICS DASHBOARD")
    st.markdown("---")

    st.markdown("### Model overview")
    st.write(
        "This application uses a trained Logistic Regression model "
        "to estimate a student's pass or fail outcome."
    )

    st.markdown("### Input variables")
    st.markdown("- **Study hours:** used by the model")
    st.markdown("- **Attendance:** recorded for display only")

    st.info(
        "The current model predicts using study hours only. "
        "Attendance does not affect the prediction."
    )

    st.markdown("---")
    st.caption("Developed by Michael Felisilda Canja")

# Main layout
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-heading">Student information</div>', unsafe_allow_html=True)
    st.markdown(
    "<div class=\"section-description\">Enter the student's details to generate a prediction.</div>",
    unsafe_allow_html=True
    )

    with st.container(border=True):
        with st.form("prediction_form"):
            student_name = st.text_input(
                "Student name",
                value="Michael Felisilda Canja",
                placeholder="Enter student name"
            )

            study_hours = st.slider(
                "Study hours per day",
                min_value=0.0,
                max_value=12.0,
                value=4.0,
                step=0.5,
                help="Select the average number of hours spent studying each day."
            )

            attendance = st.slider(
                "Attendance percentage",
                min_value=0,
                max_value=100,
                value=90,
                step=1,
                help="Record the student's attendance percentage."
            )

            st.markdown("")
            submitted = st.form_submit_button(
                "Generate prediction  →",
                use_container_width=True
            )

    if submitted:
        features = pd.DataFrame({"study_hours": [study_hours]})
        pred = int(model.predict(features)[0])
        pass_probability = float(model.predict_proba(features)[0][1]) * 100

        st.session_state["prediction_result"] = {
            "name": student_name.strip() or "Student",
            "hours": study_hours,
            "attendance": attendance,
            "pred": pred,
            "probability": pass_probability
        }

with right:
    st.markdown('<div class="section-heading">Prediction overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">Review the model output and student metrics.</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.get("prediction_result")

    with st.container(border=True):
        if result:
            passed = result["pred"] == 1
            result_class = "result-pass" if passed else "result-fail"
            result_label = "PASS" if passed else "FAIL"
            safe_name = html.escape(result["name"])

            st.markdown(
                f'<div class="{result_class}">'
                f'<div class="result-label">MODEL PREDICTED OUTCOME</div>'
                f'<div class="result-value">{result_label}</div>'
                f'<div class="result-name">{safe_name}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Study time</div>'
                    f'<div class="metric-value">{result["hours"]:g}<span style="font-size:.9rem;color:#93a4bd;"> hrs/day</span></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with metric2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Attendance</div>'
                    f'<div class="metric-value">{result["attendance"]}<span style="font-size:.9rem;color:#93a4bd;">%</span></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("#### Estimated pass probability")
            st.progress(int(round(result["probability"])))
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;gap:1rem;">'
                f'<span style="color:#93a4bd;font-size:.82rem;">Model confidence estimate</span>'
                f'<strong style="color:#45d5f2;">{result["probability"]:.1f}%</strong>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "This percentage is the model's estimated probability of the PASS class, "
                "not a guarantee of actual academic performance."
            )
        else:
            st.markdown(
                '<div class="result-empty">'
                '<div style="font-size:2rem;">📊</div>'
                '<h3 style="margin:.7rem 0 .4rem;">No prediction yet</h3>'
                '<p style="margin:0;">Complete the student information form and generate a prediction to see the results here.</p>'
                '</div>',
                unsafe_allow_html=True
            )

# Prediction chart
st.markdown("")
st.markdown('<div class="section-heading">Study hours analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Explore how the trained model classifies different study-hour values.</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    hours_range = pd.DataFrame({
        "study_hours": [x / 2 for x in range(0, 25)]
    })

    hours_range["prediction"] = model.predict(hours_range[["study_hours"]])
    hours_range["Predicted outcome"] = hours_range["prediction"].map({
        0: "Fail",
        1: "Pass"
    })

    chart_data = hours_range.set_index("study_hours")[["prediction"]].rename(
        columns={"prediction": "Predicted class (0 = Fail, 1 = Pass)"}
    )

    st.line_chart(chart_data, height=300, color="#3984ff")

    fail_count = int((hours_range["prediction"] == 0).sum())
    pass_count = int((hours_range["prediction"] == 1).sum())

    summary1, summary2 = st.columns(2)

    with summary1:
        st.metric("Study-hour values classified as FAIL", fail_count)

    with summary2:
        st.metric("Study-hour values classified as PASS", pass_count)

    st.caption(
        "The chart displays model classifications from 0 to 12 study hours per day. "
        "It illustrates the trained model's behavior and does not establish actual student performance."
    )

# Explanation and footer
st.markdown("")
with st.expander("How does this prediction work?"):
    st.write(
        "The application sends the selected study hours to a previously trained "
        "Logistic Regression model. The model returns a predicted class and an "
        "estimated probability for the PASS class. The saved model is loaded from "
        "a .joblib file, so it does not need to be retrained every time the app opens."
    )
    st.write(
        "Attendance is collected and displayed, but it is not currently a model input. "
        "To make attendance influence predictions, the training dataset and model "
        "must be updated to include it."
    )

st.markdown("""
<div class="footer">
    STUDENT PASS PREDICTOR &nbsp;·&nbsp; MACHINE LEARNING PROJECT<br>
    Educational demonstration only. Results depend on the quality and representativeness of the training data.
</div>
""", unsafe_allow_html=True)
