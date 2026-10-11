from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered",
)
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 8% 4%, #dce9ff 0, transparent 36%),
            radial-gradient(ellipse at 96% 24%, #eadfff 0, transparent 30%),
            #f7f9ff;
        color: #18233b;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 850px; padding-top: 2.5rem; padding-bottom: 3rem; }
    .hero {
        padding: 1.7rem 2rem;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(255, 255, 255, .8);
        border-radius: 24px;
        background: linear-gradient(120deg, #ffffffed, #f0edffed);
        box-shadow: 0 16px 40px #35467516;
    }
    .hero-kicker {
        color: #6551c7;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .35rem 0;
        color: #1d2b4a;
        font-size: clamp(2rem, 6vw, 3rem);
        letter-spacing: -.045em;
    }
    .hero p { margin: 0; color: #5d6982; font-size: 1.05rem; }
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        border-radius: 12px;
        border-color: #d8def0;
        background: #fff;
        color: #000;
    }
    div.stButton > button {
        min-height: 3rem;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(100deg, #5c62dc, #8053cf);
        color: white;
        font-weight: 700;
        box-shadow: 0 8px 18px #6556ca36;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 11px 22px #6556ca4a;
        color: white;
        border: 0;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e2e7f4;
        border-radius: 18px;
        background: #ffffffc9;
    }
    .result-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        font-size: 1.12rem;
        font-weight: 700;
    }
    .result-pass { color: #12664b; background: #e5f7ef; border: 1px solid #b9e9d2; }
    .result-fail { color: #934040; background: #fff0ed; border: 1px solid #f3c9c1; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Student success · prediction demo</div>
        <h1>Student Pass Predictor</h1>
        <p>Explore how study hours map to a model prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error("The trained model was not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
with st.container(border=True):
    st.subheader("Student details")
    name_column, hours_column = st.columns([1.25, 1])
    with name_column:
        student_name = st.text_input("Student name", placeholder="e.g. Alex")
    with hours_column:
        study_hours = st.number_input(
            "Study hours",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
        )

if st.button("Predict", type="primary"):
    prediction = int(model.predict([[study_hours]])[0])
    predicted_class = "Pass" if prediction == 1 else "Fail"
    name = student_name.strip() or "The student"

    result_style = "pass" if prediction == 1 else "fail"
    st.markdown(
        f'<div class="result-card result-{result_style}" role="status">'
        f'{escape(name)} is predicted to {predicted_class.lower()}.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### How this prediction was generated")
    st.write(
        f"The saved model evaluated **{study_hours:g} study hours** and returned "
        f"class **{prediction}** ({predicted_class}). It was trained using study "
        "hours only; it does not use attendance or personal student data."
    )

    hours = [value / 2 for value in range(21)]
    chart_data = pd.DataFrame(
        {
            "Study hours": hours,
            "Predicted class (0 = Fail, 1 = Pass)": [
                int(model.predict([[value]])[0]) for value in hours
            ],
        }
    )
    st.subheader("Study hours and predicted class")
    st.line_chart(chart_data, x="Study hours")
    st.caption(
        f"At {study_hours:g} study hours, the model predicts "
        f"{predicted_class.lower()} (class {prediction})."
    )
    st.caption(
        "This is a classroom demo using illustrative training data, not a real "
        "academic assessment."
    )
