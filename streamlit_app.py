import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

model = load("student_pass_model.joblib")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 28px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-pass {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 12px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #28a745;
    }
    .prediction-fail {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 12px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #dc3545;
    }
    .section-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin-top: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #1a1a2e;'>🎓 Student Pass Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Enter student details to predict academic outcomes</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("👤 Student Name", placeholder="e.g. Juan Dela Cruz")

with col2:
    hours = st.number_input("📚 Study Hours per Week", min_value=0, max_value=40, value=5, step=1)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Result"):
    prediction = model.predict(pd.DataFrame({"StudyHours": [hours]}))[0]

    if prediction == "Pass":
        st.markdown(f"<div class='prediction-pass'>✅ {name if name else 'Student'}, you are predicted to PASS!</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction-fail'>❌ {name if name else 'Student'}, you are predicted to FAIL.</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📊 View Prediction Details"):
        st.markdown(f"**Student:** {name if name else 'N/A'}")
        st.markdown(f"**Study Hours:** {hours} hours/week")
        st.markdown(f"**Prediction:** {prediction}")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🧠 How This Prediction Was Generated")
    st.write(
        "A **K-Nearest Neighbors (KNN)** model was trained on past student data "
        "containing study hours and pass/fail results, then saved to "
        "`student_pass_model.joblib`. Your study hours were compared to the "
        "closest training examples, and the majority class among those "
        "neighbors became the prediction."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📈 Study Hours vs Predicted Class")

    hours_range = list(range(1, 15))
    preds = model.predict(pd.DataFrame({"StudyHours": hours_range}))
    numeric = [1 if p == "Pass" else 0 for p in preds]
    chart_data = pd.DataFrame({
        "Study Hours": hours_range,
        "Predicted Class (1=Pass, 0=Fail)": numeric,
    })
    st.line_chart(chart_data.set_index("Study Hours"))

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Chart shows how the model's prediction changes across different study hour values.")
    st.markdown("</div>", unsafe_allow_html=True)
