import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Student Pass Predictor", page_icon="*", layout="centered")

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle { text-align: center; color: #6b7280; font-size: 1rem; margin-bottom: 2rem; }
    .result-card { padding: 1.3rem; border-radius: 14px; text-align: center;
                   font-size: 1.5rem; font-weight: 700; margin-top: 1rem; margin-bottom: 1rem; }
    .pass { background: linear-gradient(135deg, #d1fae5, #a7f3d0); color: #065f46; border: 2px solid #10b981; }
    .fail { background: linear-gradient(135deg, #fee2e2, #fecaca); color: #991b1b; border: 2px solid #ef4444; }
    .explain-box { background: #f3f4f6; border-left: 4px solid #4f46e5;
                   padding: 1rem; border-radius: 8px; font-size: 0.92rem; color: #374151; }
    .stButton>button { background: linear-gradient(90deg, #4f46e5, #06b6d4);
                       color: white; font-weight: 600; border: none;
                       padding: 0.6rem 1.2rem; border-radius: 8px; width: 100%; }
    .stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Student Pass Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Machine Learning and Streamlit</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

st.subheader("Student Information")
student_name = st.text_input("Student Name", placeholder="e.g., Juan Dela Cruz")
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0,
                              value=5.0, step=0.5,
                              help="How many hours did the student study?")

predict_clicked = st.button("Predict Result")

if predict_clicked:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = int(model.predict([[study_hours]])[0])
        probability = float(model.predict_proba([[study_hours]])[0][1])
        label = "PASS" if prediction == 1 else "FAIL"
        card_class = "pass" if prediction == 1 else "fail"

        st.markdown(
            f'<div class="result-card {card_class}">{student_name} - {label}<br>'
            f'<span style="font-size:1rem;">Pass Probability: {probability:.2%}</span></div>',
            unsafe_allow_html=True)

        st.markdown(f"""
        <div class="explain-box">
        <b>How was this prediction generated?</b><br><br>
        The saved Logistic Regression model was loaded from
        <code>student_pass_model.joblib</code>. Your input
        (<b>{study_hours} study hours</b>) was passed to the model, which
        applied the learned formula from its training data and returned
        the class with the highest probability. No retraining was performed.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Study Hours vs Predicted Class")
        hours_range = np.linspace(0, 12, 60).reshape(-1, 1)
        preds = model.predict(hours_range)
        chart_df = pd.DataFrame({
            "Study Hours": hours_range.flatten(),
            "Predicted Class": np.where(preds == 1, "Pass", "Fail")
        })
        st.scatter_chart(chart_df, x="Study Hours", y="Predicted Class",
                         color="Predicted Class", use_container_width=True)

        st.subheader("Probability of Passing")
        prob_df = pd.DataFrame({
            "Study Hours": hours_range.flatten(),
            "Pass Probability": model.predict_proba(hours_range)[:, 1]
        }).set_index("Study Hours")
        st.line_chart(prob_df, use_container_width=True)

        st.info(f"Your input: {study_hours} hours - predicted as {label} with {probability:.2%} probability.")

st.divider()
st.caption("IT Elective 3 - AI: Machine Learning | Model Persistence and Deployment Exercise")