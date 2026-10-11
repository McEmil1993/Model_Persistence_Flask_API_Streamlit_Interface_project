"""
Exercise 3 - Streamlit interface

Run with:  streamlit run streamlit_app.py
Requires student_pass_model.joblib (created by train_model.py).
"""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "student_pass_model.joblib"

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    """Load the saved model once and reuse it across reruns (no retraining)."""
    return joblib.load(MODEL_PATH)


st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f5f7ff 0%, #eef4ff 100%);
        }

        [data-testid="stAppViewContainer"] {
            padding-top: 1.5rem;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 1.5rem;
        }

        .hero-card,
        .info-card,
        .result-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(62, 100, 255, 0.12);
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 8px 24px rgba(31, 41, 55, 0.08);
        }

        .hero-card {
            margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(76, 110, 245, 0.12), rgba(61, 179, 255, 0.08));
        }

        .info-card {
            height: 100%;
        }

        .result-card {
            margin-top: 1.2rem;
            padding: 1.35rem 1.4rem;
        }

        .result-card.pass {
            border-color: rgba(32, 201, 151, 0.35);
            background: linear-gradient(135deg, rgba(32, 201, 151, 0.12), rgba(255, 255, 255, 0.9));
        }

        .result-card.fail {
            border-color: rgba(220, 53, 69, 0.28);
            background: linear-gradient(135deg, rgba(220, 53, 69, 0.10), rgba(255, 255, 255, 0.92));
        }

        .section-title {
            color: #1f2a44;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #4c6ef5, #3db3ff);
            color: white;
            font-weight: 700;
            padding: 0.7rem 1rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(76, 110, 245, 0.25);
        }

        .stMetric {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 14px;
            padding: 0.75rem;
            border: 1px solid rgba(78, 94, 125, 0.12);
        }

        div[data-testid="stFormSubmitButton"] {
            margin-top: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 Student Pass Predictor")
st.caption("Enter a student's study hours to see whether the model predicts a pass or a fail.")

if not MODEL_PATH.exists():
    st.error(f"Model file `{MODEL_PATH.name}` not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()

st.markdown("<div class='hero-card'><div class='section-title'>Student insights</div>Use the form below to estimate a student's likelihood of passing based on their study time.</div>", unsafe_allow_html=True)

with st.form("prediction_form"):
    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        student_name = st.text_input("Student name", placeholder="e.g. Maria Santos")

    with right_col:
        study_hours = st.number_input(
            "Study hours",
            min_value=0.0,
            max_value=12.0,
            value=4.0,
            step=0.5,
            help="Hours studied (0 - 12).",
        )

    submitted = st.form_submit_button("Predict", type="primary")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    if not student_name.strip():
        st.warning("Please enter the student's name first.")
        st.stop()

    predicted_class = int(model.predict([[study_hours]])[0])
    pass_probability = float(model.predict_proba([[study_hours]])[0][1])
    name = student_name.strip()

    status_class = "pass" if predicted_class == 1 else "fail"
    status_icon = "✅" if predicted_class == 1 else "⚠️"
    status_text = "PASS" if predicted_class == 1 else "FAIL"

    st.markdown(
        f"""
        <div class="result-card {status_class}">
            <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 0.3rem;">{status_icon} {name} is predicted to <strong>{status_text}</strong></div>
            <div style="font-size: 0.98rem; color: #475569;">After studying {study_hours:g} hours, the model estimates a passing chance of <strong>{pass_probability:.0%}</strong>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col, info_col = st.columns([1, 2])
    with metric_col:
        st.metric("Estimated probability of passing", f"{pass_probability:.0%}")

    with info_col:
        st.markdown("<div class='info-card'><div class='section-title'>How was this prediction generated?</div>", unsafe_allow_html=True)
        if hasattr(model, "coef_"):
            coef = float(model.coef_[0][0])
            intercept = float(model.intercept_[0])
            st.write(
                "The model is a **logistic regression** trained on past students' study hours "
                "and whether they passed. It was **loaded from a saved file**, not retrained. "
                f"It turns your input into a pass probability with the formula "
                f"`1 / (1 + e^-({coef:.2f} × hours {intercept:+.2f}))`. "
                f"For {study_hours:g} hours that gives **{pass_probability:.0%}**, "
                "and a probability of 50% or more is predicted as *pass*."
            )
        else:
            st.write(
                "The model was loaded from a saved file (not retrained) and compares your "
                "study hours with patterns learned from past students."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Bonus: chart comparing study hours with the predicted class
    st.markdown("<div class='section-title' style='margin-top: 1.2rem;'>Study hours vs. predicted class</div>", unsafe_allow_html=True)
    grid = pd.DataFrame({"study_hours": [h / 10 for h in range(0, 121)]})
    grid["pass_probability"] = model.predict_proba(grid[["study_hours"]].values)[:, 1]
    student_point = pd.DataFrame(
        {"study_hours": [study_hours], "pass_probability": [pass_probability], "student": [name]}
    )

    curve = alt.Chart(grid).mark_line().encode(
        x=alt.X("study_hours:Q", title="Study hours"),
        y=alt.Y("pass_probability:Q", title="Probability of passing", scale=alt.Scale(domain=[0, 1])),
    )
    threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[5, 5], color="gray").encode(y="y:Q")
    point = alt.Chart(student_point).mark_point(
        size=250, filled=True, color="green" if predicted_class == 1 else "red"
    ).encode(x="study_hours:Q", y="pass_probability:Q", tooltip=["student", "study_hours", "pass_probability"])

    st.altair_chart((curve + threshold + point).properties(width=700, height=360))
    st.caption(
        "The curve is the model's pass probability for every study-hours value. "
        "Above the dashed 50% line the predicted class is *pass*; below it, *fail*. "
        "The dot marks this student."
    )
