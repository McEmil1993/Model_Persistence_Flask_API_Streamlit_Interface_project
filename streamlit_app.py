from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "learnlens_model.joblib"

st.set_page_config(page_title="LearnLens", page_icon="🔎", layout="centered")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #071522 0%, #0b2234 55%, #102d42 100%);
    color: #eaf8ff;
}
.block-container { max-width: 900px; padding-top: 2rem; }
h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: #eaf8ff; }
div[data-testid="stMetric"] {
    background: #102c40;
    border: 1px solid #1e7898;
    padding: 16px;
    border-radius: 14px;
}
div[data-testid="stMetric"] label { color: #9bdcf0; }
div[data-testid="stMetricValue"] { color: #ffffff; }
.stButton > button {
    background: #00a6c7;
    color: #061522;
    border: 1px solid #56d8f0;
    border-radius: 10px;
    font-weight: 700;
}
.stButton > button:hover { background: #56d8f0; color: #061522; }
input { color: #ffffff !important; background: #18364a !important; }
div[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🔎 LearnLens")
st.write("A learning-progress demo built with a saved machine-learning model.")
st.caption("Enter study time and attendance to explore a sample progress category.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.container(border=True):
    st.subheader("Learner details")
    learner = st.text_input("Learner name", placeholder="Type a name")
    left, right = st.columns(2)
    with left:
        hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    with right:
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    submitted = st.button("Review learning progress", use_container_width=True)

if submitted:
    if not learner.strip():
        st.warning("Enter the learner's name to continue.")
    else:
        category = str(model.predict([[float(hours), float(attendance)]])[0])
        st.subheader("Learning summary")
        st.write(f"**Learner:** {learner.strip()}")
        first, second = st.columns(2)
        first.metric("Study time", f"{hours:g} hrs/day")
        second.metric("Attendance", f"{attendance:g}%")

        if category == "Strong Progress":
            st.success(f"Sample category: {category}")
        elif category == "On Track":
            st.info(f"Sample category: {category}")
        else:
            st.warning(f"Sample category: {category}")

        st.write(
            "The model uses both study hours and attendance to assign a category. "
            "It was trained on a small synthetic dataset for this class activity, "
            "so the result is a demonstration and not a real academic assessment."
        )

        chart = pd.DataFrame({
            "Measure": ["Study hours", "Attendance (%)"],
            "Value": [float(hours), float(attendance)]
        }).set_index("Measure")
        st.subheader("Input comparison")
        st.caption("These bars use different units, so compare each value with its own scale.")
        st.bar_chart(chart, horizontal=True)
        st.caption(f"Model output: {category}")

st.divider()
st.caption("LearnLens · IT Elective 3 — Machine Learning")
