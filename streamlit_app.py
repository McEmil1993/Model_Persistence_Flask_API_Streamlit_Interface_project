from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark, modern blue/cyan design
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top right, #12345a 0, #081426 38%, #050b16 100%);
    color: #eef6ff;
}
.block-container { max-width: 1100px; padding-top: 2.2rem; padding-bottom: 3rem; }
.hero {
    padding: 2rem 2.2rem; border-radius: 22px;
    background: linear-gradient(125deg, #102b4e, #0b1729 65%, #123d55);
    border: 1px solid rgba(70, 196, 255, .35);
    box-shadow: 0 18px 55px rgba(0,0,0,.25); margin-bottom: 1.5rem;
}
.eyebrow { color: #59d7ff; text-transform: uppercase; letter-spacing: .18em;
    font-size: .75rem; font-weight: 800; }
.hero h1 { color: #f4f9ff; font-size: clamp(2rem, 4vw, 3.2rem); margin: .35rem 0; }
.hero p { color: #b9cce1; font-size: 1.02rem; margin-bottom: 0; }
.panel {
    padding: 1.35rem; border-radius: 18px;
    background: rgba(14, 29, 49, .86); border: 1px solid #203b59;
}
div.stButton > button {
    width: 100%; border: 0; border-radius: 12px; padding: .75rem 1rem;
    color: #06111e; font-weight: 800;
    background: linear-gradient(90deg, #50d5ff, #4c9dff);
}
div.stButton > button:hover { color: #06111e; border: 1px solid #fff; }
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {
    background: #0b1b30; color: #f4f9ff; border-radius: 10px;
}
small, .muted { color: #a9bdd4; }
[data-testid="stMetric"] { background: #0d2037; padding: 1rem; border-radius: 14px; border: 1px solid #24435f; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = get_model()

st.markdown("""
<div class="hero">
  <div class="eyebrow">Machine Learning • Student Project</div>
  <h1>Student Pass Predictor</h1>
  <p>A simple model that estimates a student's result from study hours.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("The trained model file is missing. In this folder, run `python train_model.py` first, then refresh this page.")
    st.stop()

left, right = st.columns([1, 1.15], gap="large")
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Student details")
    student_name = st.text_input("Student name", placeholder="Enter student name")
    study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0,
                                  value=4.0, step=0.5, help="Enter a value from 0 to 24.")
    predict_clicked = st.button("⚡ Predict student result")
    st.caption("Tip: This is an educational demonstration, not a real academic assessment.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Prediction result")
    if predict_clicked:
        if not student_name.strip():
            st.warning("Please enter the student's name first.")
        else:
            prediction_number = int(model.predict([[study_hours]])[0])
            prediction = "Pass" if prediction_number == 1 else "Fail"
            probability = float(model.predict_proba([[study_hours]])[0][1])
            st.markdown(f"### {student_name.strip()}")
            if prediction == "Pass":
                st.success(f"Estimated result: **{prediction}**")
            else:
                st.error(f"Estimated result: **{prediction}**")
            m1, m2 = st.columns(2)
            m1.metric("Study hours", f"{study_hours:g} hrs")
            m2.metric("Pass probability", f"{probability:.1%}")
            st.markdown("**How was this generated?**")
            st.write(
                f"The saved Logistic Regression model received {study_hours:g} study hours "
                f"as its input and returned **{prediction}**. The model learned a pattern "
                "from the example training data in `train_model.py`."
            )
    else:
        st.info("Enter a name and study hours, then select **Predict student result**.")
        st.write("Your prediction and a short explanation will appear here.")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.subheader("Study-hours comparison")
chart_data = pd.DataFrame({
    "Study hours": [0, 1, 2, 3, 4, 5, 6, 7, 8]
})
chart_data["Predicted class (Pass = 1, Fail = 0)"] = [
    int(model.predict([[hours]])[0]) for hours in chart_data["Study hours"]
]
st.line_chart(chart_data.set_index("Study hours"))
st.caption("The chart displays predictions from the saved model for different study-hour values.")

st.markdown("""
<div class="muted" style="text-align:center; padding-top:1rem;">
  Built by Jay Mark Renos · Model Persistence & Basic Deployment
</div>
""", unsafe_allow_html=True)
