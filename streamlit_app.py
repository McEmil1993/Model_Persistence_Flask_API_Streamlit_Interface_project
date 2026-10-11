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
        :root {
            --bg-1: #0b1020;
            --bg-2: #111827;
            --panel: rgba(15, 23, 42, 0.78);
            --panel-border: rgba(148, 163, 184, 0.18);
            --accent: #7c3aed;
            --accent-2: #22d3ee;
            --text: #e2e8f0;
            --muted: #a5b4cf;
            --success: #34d399;
            --danger: #f87171;
            --shadow: rgba(15, 23, 42, 0.45);
        }

        .stApp {
            background: radial-gradient(circle at top left, rgba(124, 58, 237, 0.25), transparent 35%),
                        radial-gradient(circle at bottom right, rgba(34, 211, 238, 0.20), transparent 30%),
                        linear-gradient(160deg, var(--bg-1), var(--bg-2));
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1150px;
        }

        h1, h2, h3, p {
            color: var(--text);
        }

        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            box-shadow: 0 12px 30px var(--shadow);
            padding: 0.8rem 1rem;
        }

        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: var(--text) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.75rem 1.5rem;
            box-shadow: 0 10px 24px rgba(124, 58, 237, 0.4);
        }

        .stButton > button:hover {
            filter: brightness(1.08);
            box-shadow: 0 12px 30px rgba(34, 211, 238, 0.3);
        }

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background: rgba(15, 23, 42, 0.8);
            color: var(--text);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
        }

        .stTextInput label, .stNumberInput label,
        .stSubheader {
            color: var(--text) !important;
            font-weight: 600;
        }

        .stSuccess, .stError, .stWarning {
            border-radius: 16px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.35);
            border: 1px solid rgba(255,255,255,0.06);
        }

        .css-1d391kg {
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 Student Pass Predictor")
st.markdown(
    "<div style='padding: 1.1rem 1.2rem; margin-bottom: 1rem; border: 1px solid rgba(148,163,184,0.16); background: rgba(15, 23, 42, 0.52); border-radius: 18px; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.35);'>"
    "<span style='display: inline-block; font-size: 0.85rem; letter-spacing: 0.09em; text-transform: uppercase; color: #7dd3fc; font-weight: 700;'>Academic forecast</span>"
    "<div style='margin-top: 0.4rem; font-size: 1.02rem; color: #dbeafe;'>Enter a student's study hours to see whether the model predicts a pass or a fail.</div>"
    "</div>",
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"Model file `{MODEL_PATH.name}` not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
with st.form("prediction_form"):
    student_name = st.text_input("Student name", placeholder="e.g. Maria Santos")
    study_hours = st.number_input(
        "Study hours", min_value=0.0, max_value=12.0, value=4.0, step=0.5,
        help="Hours studied (0 - 12).",
    )
    submit = st.form_submit_button("Predict")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submit:
    if not student_name.strip():
        st.warning("Please enter the student's name first.")
        st.stop()

    predicted_class = int(model.predict([[study_hours]])[0])
    pass_probability = float(model.predict_proba([[study_hours]])[0][1])
    name = student_name.strip()

    col_metric, col_status = st.columns([1, 2])
    with col_metric:
        st.metric("Estimated probability of passing", f"{pass_probability:.0%}")
    with col_status:
        if predicted_class == 1:
            st.success(f"✅ **{name}** is predicted to **PASS** after studying {study_hours:g} hours.")
        else:
            st.error(f"⚠️ **{name}** is predicted to **FAIL** with {study_hours:g} hours of study.")

    # Short explanation of how the prediction was generated
    st.subheader("How was this prediction generated?")
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

    # Bonus: chart comparing study hours with the predicted class
    st.subheader("Study hours vs. predicted class")
    grid = pd.DataFrame({"study_hours": [h / 10 for h in range(0, 121)]})
    grid["pass_probability"] = model.predict_proba(grid[["study_hours"]].values)[:, 1]
    student_point = pd.DataFrame(
        {"study_hours": [study_hours], "pass_probability": [pass_probability], "student": [name]}
    )

    curve = alt.Chart(grid).mark_line(color="#7c3aed", strokeWidth=3).encode(
        x=alt.X("study_hours:Q", title="Study hours"),
        y=alt.Y("pass_probability:Q", title="Probability of passing", scale=alt.Scale(domain=[0, 1])),
    )
    threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[5, 5], color="#8ecae6").encode(y="y:Q")
    point = alt.Chart(student_point).mark_point(
        size=160, filled=True, color="#22d3ee" if predicted_class == 1 else "#f87171"
    ).encode(x="study_hours:Q", y="pass_probability:Q", tooltip=["student", "study_hours", "pass_probability"])

    chart = (curve + threshold + point).properties(width=700, height=340)
    chart = chart.configure(background="rgba(0,0,0,0)")
    chart = chart.configure_axis(labelColor="#dbeafe", titleColor="#e2e8f0")
    chart = chart.configure_legend(labelColor="#dbeafe", titleColor="#e2e8f0")
    chart = chart.configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "The curve is the model's pass probability for every study-hours value. "
        "Above the dashed 50% line the predicted class is *pass*; below it, *fail*. "
        "The dot marks this student."
    )
