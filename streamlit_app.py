import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="*",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #4f46e5, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .stTextInput label, .stNumberInput label {
        font-weight: 600 !important;
        color: #374151 !important;
        font-size: 1rem !important;
    }
    .stTextInput input, .stNumberInput input {
        border-radius: 10px !important;
        border: 2px solid #e5e7eb !important;
        padding: 0.6rem !important;
        font-size: 1rem !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 0.85rem 1.5rem !important;
        border-radius: 12px !important;
        width: 100% !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.55) !important;
    }
    .result-card {
        padding: 1.8rem;
        border-radius: 18px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    .pass {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        color: #065f46;
        border: 3px solid #10b981;
    }
    .fail {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
        border: 3px solid #ef4444;
    }
    .explain-box {
        background: white;
        border-left: 5px solid #4f46e5;
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        font-size: 0.95rem;
        color: #374151;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
        line-height: 1.7;
    }
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }
    h3 {
        color: #1f2937 !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    .caption-text {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 2rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Student Pass Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Machine Learning and Streamlit</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

st.markdown("### Student Information")

student_name = st.text_input(
    "Student Name",
    placeholder="e.g., Juan Dela Cruz"
)

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
    help="How many hours did the student study?"
)

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
            f'<div class="result-card {card_class}">'
            f'{student_name} -> {label}<br>'
            f'<span style="font-size:1rem; font-weight:600;">'
            f'Pass Probability: {probability:.2%}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="explain-box">
        <b>How was this prediction generated?</b><br><br>
        The saved <b>Logistic Regression</b> model was loaded from
        <code>student_pass_model.joblib</code>. Your input
        (<b>{study_hours} study hours</b>) was passed to the model, which
        applied the learned formula from its training data and returned
        the class with the highest probability.
        <br><br>
        <i>No retraining was performed - the model was reused from disk.</i>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Study Hours vs Predicted Class")

        hours_range = np.linspace(0, 12, 60).reshape(-1, 1)
        preds = model.predict(hours_range)

        chart_df = pd.DataFrame({
            "Study Hours": hours_range.flatten(),
            "Predicted Class": np.where(preds == 1, "Pass", "Fail")
        })

        st.scatter_chart(
            chart_df,
            x="Study Hours",
            y="Predicted Class",
            color="Predicted Class",
            use_container_width=True
        )

        st.markdown("### Probability of Passing")

        prob_df = pd.DataFrame({
            "Study Hours": hours_range.flatten(),
            "Pass Probability": model.predict_proba(hours_range)[:, 1]
        }).set_index("Study Hours")

        st.line_chart(prob_df, use_container_width=True)

        st.info(f"Your input: {study_hours} hours - predicted as {label} with {probability:.2%} probability.")

st.markdown(
    '<div class="caption-text">'
    'IT Elective 3 - AI: Machine Learning | Model Persistence and Deployment Exercise'
    '</div>',
    unsafe_allow_html=True
)