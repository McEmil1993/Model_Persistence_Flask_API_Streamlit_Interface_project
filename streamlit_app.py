from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Predictor", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root {
        --ink: #203c35;
        --muted: #62766e;
        --paper: #f4f6ef;
        --green: #2f7058;
        --coral: #d36e53;
        --line: #dce4d9;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1100px; padding-top: 2.5rem; }
    html, body, [class*="st-"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { color: var(--ink); }
    h1 { font-family: 'Playfair Display', Georgia, serif !important; font-size: 2.7rem !important; }
    .eyebrow { color: var(--green); text-transform: uppercase; font-size: .75rem; font-weight: 700; letter-spacing: .12em; }
    .subhead { color: var(--muted); font-size: 1rem; margin-top: -.6rem; }
    div[data-testid="stForm"] { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 1.25rem 1.4rem; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button { background: var(--green); color: white; border: 0; border-radius: 5px; font-weight: 700; min-height: 2.8rem; }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { background: #245b46; color: white; }
    .result-panel { border-left: 4px solid var(--coral); background: #fff; padding: 1.2rem 1.4rem; border-radius: 0 8px 8px 0; }
    .result-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; font-weight: 700; }
    .result-value { color: var(--ink); font-family: 'Playfair Display', Georgia, serif; font-size: 1.7rem; font-weight: 700; }
    .fine-print { color: var(--muted); font-size: .88rem; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.markdown('<div class="eyebrow">Learning Lab / Student Outcomes</div>', unsafe_allow_html=True)
st.title("Student Pass Predictor")
st.markdown(
    '<p class="subhead">Explore how study time relates to the model\'s pass prediction.</p>',
    unsafe_allow_html=True,
)
st.divider()

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run `python train_model.py` and refresh this page.")
    st.stop()

model = load_model()

input_column, result_column = st.columns([0.9, 1.1], gap="large")
with input_column:
    st.subheader("Student details")
    with st.form("prediction_form"):
        student_name = st.text_input("Student name", placeholder="e.g. Alex Morgan")
        study_hours = st.number_input(
            "Study hours per day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
        )
        submitted = st.form_submit_button("Predict outcome", use_container_width=True)

if submitted:
    st.session_state["student_name"] = student_name.strip() or "Student"
    st.session_state["study_hours"] = float(study_hours)
    st.session_state["prediction"] = int(model.predict([[study_hours]])[0])

with result_column:
    st.subheader("Prediction")
    if "prediction" not in st.session_state:
        st.info("Enter a name and study schedule to see a prediction.")
    else:
        name = st.session_state["student_name"]
        hours = st.session_state["study_hours"]
        prediction = st.session_state["prediction"]
        outcome = "Pass" if prediction else "Not yet"
        if prediction:
            st.success(f"{name}: predicted to pass")
        else:
            st.warning(f"{name}: not predicted to pass yet")
        st.markdown(
            f'<div class="result-panel"><div class="result-label">Model result</div>'
            f'<div class="result-value">{outcome}</div>'
            f'<p class="fine-print">Based on {hours:g} study hours per day.</p></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "The classifier was trained on example study-hour and pass/fail records. "
            "It uses study hours only; attendance is not part of this model."
        )

        chart_hours = sorted(set([float(hour) for hour in range(13)] + [hours]))
        chart_predictions = model.predict([[hour] for hour in chart_hours])
        chart_data = pd.DataFrame(
            {"Study hours": chart_hours, "Predicted class (0 = Not yet, 1 = Pass)": chart_predictions}
        )
        st.markdown("#### Study hours and predicted class")
        st.line_chart(
            chart_data,
            x="Study hours",
            y="Predicted class (0 = Not yet, 1 = Pass)",
            height=250,
        )