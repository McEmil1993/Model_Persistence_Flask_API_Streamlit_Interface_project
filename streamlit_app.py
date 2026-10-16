from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "skilltrack_model.joblib"

st.set_page_config(page_title="SkillTrack ML", page_icon="📈", layout="centered")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0b1110 0%, #10221f 58%, #12352f 100%);
    color: #eefcf6;
}
.block-container { max-width: 900px; padding-top: 2rem; }
h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: #eefcf6; }
div[data-testid="stMetric"] {
    background: #142b27;
    border: 1px solid #286e5c;
    padding: 16px;
    border-radius: 14px;
}
div[data-testid="stMetric"] label { color: #a8d9c8; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff; }
.stButton > button {
    background: #159b78;
    color: white;
    border: 1px solid #52d6ac;
    border-radius: 10px;
    font-weight: 600;
}
.stButton > button:hover { background: #22b58d; color: #07130f; }
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    color: #ffffff !important;
    background-color: #1a302c !important;
}
div[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📈 SkillTrack ML")
st.write("A learning-progress demonstration powered by a saved machine-learning model.")
st.caption("Enter a learner's study routine to explore a sample progress category.")

if not MODEL_PATH.exists():
    st.error("The saved model is missing. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.container(border=True):
    st.subheader("Learner profile")
    learner_name = st.text_input("Learner name", placeholder="Type a name")
    left, right = st.columns(2)
    with left:
        study_hours = st.number_input(
            "Study hours per day", min_value=0.0, max_value=24.0,
            value=5.0, step=0.5
        )
    with right:
        attendance = st.number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0,
            value=75.0, step=1.0
        )
    submitted = st.button("Review learning progress", use_container_width=True)

if submitted:
    if not learner_name.strip():
        st.warning("Enter a learner name to continue.")
    else:
        progress = str(model.predict([[float(study_hours), float(attendance)]])[0])
        st.subheader("Progress overview")
        st.write(f"**Learner:** {learner_name.strip()}")
        a, b = st.columns(2)
        a.metric("Study time", f"{study_hours:g} hrs/day")
        b.metric("Attendance", f"{attendance:g}%")

        if progress == "Ready to Advance":
            st.success(f"Sample category: {progress}")
            advice = "Keep the routine consistent and try a new learning goal."
        elif progress == "Developing":
            st.info(f"Sample category: {progress}")
            advice = "Build a steady study schedule and review topics regularly."
        else:
            st.warning(f"Sample category: {progress}")
            advice = "Try shorter focused study sessions and seek help on difficult topics."

        st.write(advice)
        st.caption(
            "This model uses a small synthetic classroom dataset. Its output is "
            "for demonstrating model persistence and prediction, not for grading "
            "or making a reliable judgement about a real learner."
        )

        chart_data = pd.DataFrame({
            "Input measure": ["Study hours per day", "Attendance (%)"],
            "Value": [float(study_hours), float(attendance)]
        }).set_index("Input measure")
        st.subheader("Input comparison")
        st.caption("The bars use different units; compare each value with its own scale.")
        st.bar_chart(chart_data, horizontal=True)
        st.caption(f"Model output: {progress} (demonstration label only).")

st.divider()
st.caption("SkillTrack ML · IT Elective 3 — Machine Learning")
