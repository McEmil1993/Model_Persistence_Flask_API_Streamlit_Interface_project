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

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* App background */
        .stApp { background: linear-gradient(180deg, #f6f8fc 0%, #eef1f7 100%); }

        /* Hero header */
        .hero {
            padding: 2rem 2.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #4f6df5 0%, #7b5cf0 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(79, 109, 245, 0.25);
        }
        .hero h1 { margin: 0; font-size: 2.2rem; }
        .hero p  { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1.05rem; }

        /* Cards */
        .card {
            background: white;
            padding: 1.4rem 1.6rem;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(20, 30, 60, 0.07);
            margin-bottom: 1.2rem;
        }

        /* Result banners */
        .result-pass, .result-fail {
            padding: 1.1rem 1.4rem;
            border-radius: 12px;
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .result-pass { background: #e7f7ec; color: #1d7a3d; border: 1px solid #b7e4c7; }
        .result-fail { background: #fdecec; color: #b03030; border: 1px solid #f5c2c2; }

        /* Section titles */
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #2a2f4a;
            margin: 0.4rem 0 0.6rem 0;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] { background: #232946; }
        section[data-testid="stSidebar"] * { color: #eef0fa !important; }
        section[data-testid="stSidebar"] input { color: #232946 !important; }

        /* Primary button */
        div.stButton > button[kind="primary"] {
            width: 100%;
            border-radius: 10px;
            padding: 0.6rem 1rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the saved model once and reuse it across reruns (no retraining)."""
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Pass Predictor</h1>
        <p>Enter a student's study hours and see whether the model predicts a pass or a fail.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"Model file `{MODEL_PATH.name}` not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()

# ---------------------------------------------------------------------------
# Inputs (sidebar)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📝 Student details")
    student_name = st.text_input("Student name", placeholder="e.g. Maria Santos")
    study_hours = st.slider(
        "Study hours",
        min_value=0.0, max_value=12.0, value=4.0, step=0.5,
        help="Hours studied (0 - 12).",
    )
    st.caption(f"Selected: **{study_hours:g} hours**")
    predict_clicked = st.button("🔮 Predict", type="primary")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_clicked:
    if not student_name.strip():
        st.warning("Please enter the student's name first.")
        st.stop()

    predicted_class = int(model.predict([[study_hours]])[0])
    pass_probability = float(model.predict_proba([[study_hours]])[0][1])
    name = student_name.strip()

    # ---- Result row: banner + metric --------------------------------------
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
        if predicted_class == 1:
            st.markdown(
                f'<div class="result-pass">✅ <b>{name}</b> is predicted to <b>PASS</b> '
                f"after studying {study_hours:g} hours.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-fail">⚠️ <b>{name}</b> is predicted to <b>FAIL</b> '
                f"with {study_hours:g} hours of study.</div>",
                unsafe_allow_html=True,
            )
        st.progress(pass_probability)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confidence</div>', unsafe_allow_html=True)
        st.metric("Estimated probability of passing", f"{pass_probability:.0%}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Explanation -------------------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 How was this prediction generated?</div>', unsafe_allow_html=True)
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

    # ---- Chart -------------------------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Study hours vs. predicted class</div>', unsafe_allow_html=True)

    grid = pd.DataFrame({"study_hours": [h / 10 for h in range(0, 121)]})
    grid["pass_probability"] = model.predict_proba(grid[["study_hours"]].values)[:, 1]
    student_point = pd.DataFrame(
        {"study_hours": [study_hours], "pass_probability": [pass_probability], "student": [name]}
    )

    curve = alt.Chart(grid).mark_line(strokeWidth=3, color="#4f6df5").encode(
        x=alt.X("study_hours:Q", title="Study hours"),
        y=alt.Y("pass_probability:Q", title="Probability of passing",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
    )
    threshold = (
        alt.Chart(pd.DataFrame({"y": [0.5]}))
        .mark_rule(strokeDash=[6, 4], color="#999", strokeWidth=1.5)
        .encode(y="y:Q")
    )
    point = (
        alt.Chart(student_point)
        .mark_point(size=260, filled=True, color="#1d7a3d" if predicted_class == 1 else "#c0392b",
                    stroke="white", strokeWidth=2)
        .encode(x="study_hours:Q", y="pass_probability:Q",
                tooltip=["student", "study_hours", "pass_probability"])
    )
    st.altair_chart((curve + threshold + point).properties(width=600, height=320))
    st.caption(
        "The curve is the model's pass probability for every study-hours value. "
        "Above the dashed 50% line the predicted class is *pass*; below it, *fail*. "
        "The dot marks this student."
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Empty state before the first prediction
    st.markdown(
        """
        <div class="card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem;">👈</div>
            <div class="section-title" style="text-align:center;">Ready when you are</div>
            <p style="color:#6b7280;">Fill in the student's details in the sidebar and hit
            <b>Predict</b> to see the result here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )