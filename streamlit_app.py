import os
import joblib
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

# Light custom styling
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 800px; }
    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        font-size: 1.1rem;
    }
    .pass-card { background-color: #e8f8ee; border-left: 6px solid #22a55b; }
    .fail-card { background-color: #fdecec; border-left: 6px solid #d64545; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "student_pass_model.joblib")
    return joblib.load(path)


model = load_model()

# ---------- Header ----------
st.title("🎓 Student Pass Predictor")
st.caption("Enter a student's name and study hours to predict whether they will pass.")

# ---------- Inputs ----------
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        student_name = st.text_input("Student name", placeholder="e.g., Juan Dela Cruz")
    with col2:
        study_hours = st.number_input(
            "Study hours per day",
            min_value=0.0, max_value=24.0, value=5.0, step=0.5
        )

    predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

# ---------- Prediction ----------
if predict_clicked:
    name = student_name.strip() or "The student"

    X_new = pd.DataFrame([[study_hours]], columns=["study_hours"])
    prediction = int(model.predict(X_new)[0])
    probability = model.predict_proba(X_new)[0][1]  # chance of passing

    if prediction == 1:
        st.markdown(
            f"<div class='result-card pass-card'>✅ <b>{name}</b> is <b>likely to PASS</b> "
            f"(chance: {probability:.0%}).</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='result-card fail-card'>❌ <b>{name}</b> is <b>likely to FAIL</b> "
            f"(chance of passing: {probability:.0%}).</div>",
            unsafe_allow_html=True,
        )

    # Metrics
    m1, m2 = st.columns(2)
    m1.metric("Study hours", f"{study_hours:g} h/day")
    m2.metric("Pass probability", f"{probability:.0%}")

    # How the prediction was generated
    with st.expander("How was this prediction generated?"):
        st.write(
            "The model was trained on past student data that linked study hours to pass/fail results. "
            "It calculates a probability of passing based on the number of study hours entered. "
            "If the probability is 50% or higher, the student is predicted to **pass**; otherwise, **fail**."
        )

    # ---------- Bonus chart ----------
    st.subheader("Study hours vs. predicted result")

    hours_range = pd.DataFrame({"study_hours": [h / 2 for h in range(0, 49)]})  # 0 to 24 h
    hours_range["probability"] = model.predict_proba(hours_range[["study_hours"]])[:, 1]
    hours_range["predicted_class"] = model.predict(hours_range[["study_hours"]])
    hours_range["predicted_class"] = hours_range["predicted_class"].map({1: "Pass", 0: "Fail"})

    curve = (
        alt.Chart(hours_range)
        .mark_line(color="#4c78a8", strokeWidth=3)
        .encode(
            x=alt.X("study_hours:Q", title="Study hours per day"),
            y=alt.Y("probability:Q", title="Probability of passing", scale=alt.Scale(domain=[0, 1])),
        )
    )

    points = (
        alt.Chart(hours_range)
        .mark_circle(size=40, opacity=0.6)
        .encode(
            x="study_hours:Q",
            y="probability:Q",
            color=alt.Color(
                "predicted_class:N",
                scale=alt.Scale(domain=["Pass", "Fail"], range=["#22a55b", "#d64545"]),
                title="Predicted",
            ),
            tooltip=["study_hours", alt.Tooltip("probability:Q", format=".0%"), "predicted_class"],
        )
    )

    student_point = (
        alt.Chart(pd.DataFrame({"study_hours": [study_hours], "probability": [probability]}))
        .mark_point(size=250, color="black", filled=True, shape="diamond")
        .encode(x="study_hours:Q", y="probability:Q", tooltip=["study_hours", "probability"])
    )

    threshold = (
        alt.Chart(pd.DataFrame({"y": [0.5]}))
        .mark_rule(strokeDash=[6, 4], color="gray")
        .encode(y="y:Q")
    )

    st.altair_chart((curve + points + student_point + threshold).properties(height=320),
                    use_container_width=True)
    st.caption("◆ = this student's input. Dashed line = 50% pass threshold.")