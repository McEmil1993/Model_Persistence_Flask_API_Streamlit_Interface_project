"""Exercise 3 - Streamlit interface for the student pass model."""
import math

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

THRESHOLD = 0.5
TRAINED_MAX_HOURS = 9  # the sample training data goes up to 9 study hours

st.markdown(
    """
    <style>
    .stApp { background-color: #0f1117; }
    .stApp, .stApp p, .stApp label, [data-testid="stWidgetLabel"] { color: #f1f5f9 !important; }
    [data-testid="stCaptionContainer"] { color: #94a3b8 !important; }
    div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {
        background-color: #1a1d27 !important;
        border-color: #2d3240 !important;
    }
    input { color: #f1f5f9 !important; }

    .st-key-study_hours div[data-baseweb="input"] > div,
    .st-key-study_hours div[data-baseweb="base-input"],
    .st-key-study_hours input {
        background-color: #000000 !important;
    }
    .st-key-study_hours button {
        color: #000000 !important;
        border: none !important;
        box-shadow: none !important;
    }
    .st-key-study_hours button:first-of-type {
        background-color: #fca5a5 !important;
    }
    .st-key-study_hours button:last-of-type {
        background-color: #93c5fd !important;
    }
    .st-key-study_hours button svg {
        fill: #000000 !important;
        color: #000000 !important;
    }

    .app-header {
        background: linear-gradient(135deg, #30e9fe 0%, #0d0552 100%);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
    }
    .app-header h1 { color: #ffffff; margin: 0; font-size: 2.1rem; font-weight: 700; }
    .app-header p { color: rgba(255, 255, 255, 0.92); margin: 0.4rem 0 0 0; }

    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3240;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-label { color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.25rem; }
    .metric-value { font-size: 1.8rem; font-weight: 600; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1a1d27;
        border-radius: 14px;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.25rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: #ffffff;
        transform: translateY(-1px);
    }
    div.stButton > button:focus { color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label, value, color="#f1f5f9"):
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div></div>'
    )


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

# This page only has study hours as an input. Stop with a clear message
# if the saved model was trained with attendance too.
if "attendance_percentage" in model.feature_names_in_:
    st.error(
        "The saved model needs attendance. Set USE_ATTENDANCE = False in "
        "train_model.py, run it again, then restart Streamlit."
    )
    st.stop()

st.markdown(
    '<div class="app-header"><h1>🎓 Student Pass Predictor</h1>'
    '<p>Predict whether a student will pass, based on study hours.</p></div>',
    unsafe_allow_html=True,
)

# ----- Inputs -----
name = st.text_input("Student name")
study_hours = st.number_input(
    "Study hours per day", min_value=0.0, max_value=24.0, value=0.0, step=0.5, key="study_hours"
)
predict_clicked = st.button("Predict")

# ----- Results -----
if not predict_clicked:
    st.info("Enter the student's name and study hours, then click Predict.")
elif not name.strip():
    st.warning("Please enter the student's name.")
elif study_hours is None:
    st.warning("Please enter the student's study hours.")
else:
    features = pd.DataFrame([{"study_hours": study_hours}])[model.feature_names_in_]

    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])
    label = "Pass" if pred == 1 else "Fail"
    class_color = "#34d399" if pred == 1 else "#f87171"

    if pred == 1:
        st.success(f"**{name} is predicted to PASS**\n\nPass probability: {prob:.0%}")
    else:
        st.error(f"**{name} is predicted to FAIL**\n\nPass probability: {prob:.0%}")

    if study_hours > TRAINED_MAX_HOURS:
        st.info(
            f"The model was trained on up to {TRAINED_MAX_HOURS} study hours, "
            "so results above that are extrapolated."
        )

    cards = [
        ("Study hours", f"{study_hours:g} h", "#f1f5f9"),
        ("Pass probability", f"{prob:.0%}", "#f1f5f9"),
        ("Predicted class", label, class_color),
    ]
    row = "".join(
        '<div style="flex:1;min-width:0;background:#1a1d27;border:1px solid #2d3240;'
        'border-radius:12px;padding:0.6rem 0.9rem;">'
        f'<div style="color:#94a3b8;font-size:0.75rem;">{title}</div>'
        f'<div style="color:{color};font-size:1.25rem;font-weight:600;">{value}</div>'
        "</div>"
        for title, value, color in cards
    )
    st.markdown(
        f'<div style="display:flex;flex-direction:row;gap:0.6rem;flex-wrap:nowrap;">{row}</div>',
        unsafe_allow_html=True,
    )

    st.write(f"Confidence of passing: {prob:.0%} (threshold {THRESHOLD:.0%})")
    st.progress(prob)

    # ----- How it was predicted (box, built from the study hours entered) -----
    intercept = float(model.intercept_[0])
    weights = model.coef_[0]
    row = features.iloc[0]

    terms = " + ".join(f"{w:.3f} × {row[f]:g}" for w, f in zip(weights, model.feature_names_in_))
    score = intercept + sum(w * row[f] for w, f in zip(weights, model.feature_names_in_))
    calc_prob = 1 / (1 + math.exp(-score))
    weight_lines = "\n".join(f"- Weight for {f} = {w:.3f}" for w, f in zip(weights, model.feature_names_in_))
    comparison = "at or above" if calc_prob >= THRESHOLD else "below"

    explanation = f"""
The model is a **logistic regression** trained on past student data.

**Step 1: Input.** Study hours = {study_hours:g} h.

**Step 2: Score.** Each input is multiplied by its learned weight, and the intercept is added.

- Intercept = {intercept:.3f}
{weight_lines}
- Score = {intercept:.3f} + {terms} = **{score:.3f}**

**Step 3: Probability.** The score is converted to a probability with 1 ÷ (1 + e^(−score)) = 1 ÷ (1 + e^(−{score:.3f})) = **{calc_prob:.1%}**.

**Step 4: Decision.** {calc_prob:.1%} is {comparison} the {THRESHOLD:.0%} threshold, so the model predicts **{label}**.
"""

    with st.container(border=True):
        st.subheader("How it was predicted")
        st.markdown(explanation)

    # ----- Hours vs. class (box) -----
    hours_range = [h / 2 for h in range(0, 49)]  # 0 to 24 hours
    curve = pd.DataFrame({"study_hours": hours_range})
    curve["pass_probability"] = model.predict_proba(curve[["study_hours"]])[:, 1]

    x = alt.X(
        "study_hours:Q",
        title="Study hours",
        scale=alt.Scale(domain=[0, 24]),
        axis=alt.Axis(values=list(range(0, 25, 2))),
    )
    y = alt.Y(
        "pass_probability:Q",
        title="Pass probability",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format=".0%", values=[0, 0.2, 0.4, 0.6, 0.8, 1]),
    )

    student = pd.DataFrame({
        "study_hours": [study_hours],
        "pass_probability": [prob],
        "result": [label],
    })

    chart = (
        alt.layer(
            alt.Chart(curve).mark_line(color="#4169e1", strokeWidth=3).encode(x=x, y=y),
            alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(
                color="#94a3b8", strokeDash=[5, 5]
            ).encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1]), axis=None)),
            alt.Chart(student).mark_circle(size=220, stroke="#0f1117", strokeWidth=2).encode(
                x=x,
                y=y,
                color=alt.Color(
                    "result:N",
                    scale=alt.Scale(domain=["Pass", "Fail"], range=["#34d399", "#f87171"]),
                    legend=alt.Legend(title="Predicted class"),
                ),
                tooltip=[
                    alt.Tooltip("study_hours:Q", title="Study hours"),
                    alt.Tooltip("pass_probability:Q", title="Pass probability", format=".0%"),
                    alt.Tooltip("result:N", title="Predicted class"),
                ],
            ),
        )
        .properties(height=320, background="transparent")
        .configure_axis(labelColor="#94a3b8", titleColor="#e2e8f0",
                        gridColor="#2d3240", domainColor="#2d3240")
        .configure_legend(labelColor="#e2e8f0", titleColor="#e2e8f0")
        .configure_view(strokeWidth=0)
    )

    with st.container(border=True):
        st.subheader("Hours vs. class")
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"{name} studies {study_hours:g} h → predicted class: {label}. "
            f"The dashed line is the {THRESHOLD:.0%} decision threshold."
        )