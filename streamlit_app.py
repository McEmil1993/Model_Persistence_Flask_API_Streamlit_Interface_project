from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="wide",
)
st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp {
        --text-color: #18233b;
        background:
            radial-gradient(ellipse at 4% 0%, #dce9ff 0, transparent 34%),
            radial-gradient(ellipse at 95% 18%, #efe2ff 0, transparent 30%),
            #f6f8fc;
        color: #18233b;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: clamp(1.5rem, 4vw, 2.8rem);
        margin-bottom: 1.4rem;
        border: 1px solid #ffffffc9;
        border-radius: 26px;
        background: linear-gradient(115deg, #ffffffed 0%, #f0edffed 100%);
        box-shadow: 0 18px 50px #283c6817;
    }
    .hero:after {
        content: "✦";
        position: absolute;
        top: -2.7rem;
        right: 2rem;
        color: #725fe31a;
        font-size: 12rem;
        line-height: 1;
    }
    .hero-kicker {
        color: #6551c7;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .hero h1 {
        position: relative;
        z-index: 1;
        margin: .45rem 0 .55rem;
        color: #1d2b4a;
        font-size: clamp(2rem, 5vw, 3.35rem);
        letter-spacing: -.05em;
    }
    .hero p { max-width: 650px; margin: 0; color: #5d6982; font-size: 1.05rem; }
    .section-label {
        margin: .2rem 0 .7rem;
        color: #6653bd;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        min-height: 2.9rem;
        border-radius: 12px;
        border-color: #d8def0;
        background: #fff;
        color: #18233b;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stTextInput"] label,
    [data-testid="stNumberInput"] label {
        color: #34415b !important;
    }
    div.stButton > button {
        min-height: 3.1rem;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(100deg, #5c62dc, #8053cf);
        color: white;
        font-weight: 750;
        box-shadow: 0 8px 18px #6556ca36;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 11px 22px #6556ca4a;
        color: white;
        border: 0;
    }
    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 3.1rem;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(100deg, #5c62dc, #8053cf);
        color: #fff;
        font-weight: 750;
        box-shadow: 0 8px 18px #6556ca36;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        color: #fff;
        border: 0;
        background: linear-gradient(100deg, #5057cd, #7045bf);
    }
    [data-testid="stAlert"] p { color: #42516d !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e2e7f4;
        border-radius: 19px;
        background: #ffffffd9;
        box-shadow: 0 10px 28px #34456b0b;
    }
    .result-card {
        padding: 1.2rem 1.35rem;
        border-radius: 16px;
        font-size: 1.18rem;
        font-weight: 750;
    }
    .result-pass { color: #12664b; background: #e5f7ef; border: 1px solid #b9e9d2; }
    .result-fail { color: #934040; background: #fff0ed; border: 1px solid #f3c9c1; }
    .fine-print { color: #737e94; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">A small machine-learning classroom demo</div>
        <h1>Student Pass Predictor <span aria-hidden="true">🎓</span></h1>
        <p>Enter a student name and study time to explore what the saved model predicts.
        Clear, quick, and designed for learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error("The trained model was not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
form_column, info_column = st.columns([1.15, 0.85], gap="large")

with form_column:
    with st.container(border=True):
        st.markdown('<div class="section-label">01 · Student details</div>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            student_name = st.text_input("Student name", placeholder="e.g. Alex Morgan")
            study_hours = st.number_input(
                "Study hours",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="Choose a value from 0 to 10 hours.",
            )
            submitted = st.form_submit_button("✨  Generate prediction", type="primary", use_container_width=True)

with info_column:
    with st.container(border=True):
        st.markdown('<div class="section-label">02 · About this demo</div>', unsafe_allow_html=True)
        st.subheader("A saved model, ready to use")
        st.write(
            "The trained model is loaded from a local file. Entering a study-hours "
            "value asks that model for a prediction—no retraining is needed."
        )
        st.info(
            "This classroom model uses study hours only. Attendance is not part of "
            "its prediction, and this result is not a real academic assessment.",
            icon="💡",
        )

if submitted:
    prediction = int(model.predict([[study_hours]])[0])
    predicted_class = "Pass" if prediction == 1 else "Fail"
    name = student_name.strip() or "The student"
    result_style = "pass" if prediction == 1 else "fail"

    st.markdown('<div class="section-label">03 · Your result</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-card result-{result_style}" role="status">'
        f'{escape(name)} is predicted to {predicted_class.lower()}.</div>',
        unsafe_allow_html=True,
    )

    metric_name, metric_hours, metric_class = st.columns(3)
    metric_name.metric("Student", name)
    metric_hours.metric("Study time", f"{study_hours:g} hours")
    metric_class.metric("Predicted class", predicted_class, f"Class {prediction}")

    chart_column, explanation_column = st.columns([1.15, 0.85], gap="large")
    with chart_column:
        with st.container(border=True):
            st.subheader("Study hours vs. predicted class")
            hours = [value / 2 for value in range(21)]
            chart_data = pd.DataFrame(
                {
                    "Study hours": hours,
                    "Model prediction (0 = Fail, 1 = Pass)": [
                        int(model.predict([[value]])[0]) for value in hours
                    ],
                }
            )
            st.line_chart(
                chart_data,
                x="Study hours",
                y="Model prediction (0 = Fail, 1 = Pass)",
                height=280,
            )
            st.caption(
                f"At {study_hours:g} study hours, the model returns "
                f"{predicted_class.lower()} (class {prediction})."
            )

    with explanation_column:
        with st.container(border=True):
            st.subheader("How was this generated?")
            st.write(
                f"The saved model evaluated **{study_hours:g} study hours** and "
                f"returned class **{prediction}** (**{predicted_class}**). "
                "Its learned rule was stored in `student_pass_model.joblib` "
                "after training."
            )
            st.markdown(
                '<p class="fine-print">The example data and prediction are for '
                "learning purposes only—not for evaluating real students.</p>",
                unsafe_allow_html=True,
            )
