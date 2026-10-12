import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="🎓", layout="centered")

# Simple custom styling
st.markdown("""
<style>
.main-title {text-align:center; font-size:2.4rem; font-weight:800; margin-bottom:0;}
.subtitle {text-align:center; color:gray; margin-bottom:1.5rem;}
div.stButton > button {width:100%; background:#4F46E5; color:white; border-radius:10px; height:3rem; font-weight:600;}
div.stButton > button:hover {background:#4338CA; color:white;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎓 Student Pass/Fail Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter study hours and let the trained model predict the result.</p>', unsafe_allow_html=True)

model = joblib.load("student_pass_model.joblib")

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("👤 Student name", placeholder="e.g., Gelyn")
    with col2:
        hours = st.number_input("📚 Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    predict = st.button("🔮 Predict")

if predict:
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        X = pd.DataFrame({"study_hours": [hours]})
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        if pred == 1:
            st.success(f"✅ {name} is predicted to PASS.")
        else:
            st.error(f"❌ {name} is predicted to FAIL.")

        m1, m2 = st.columns(2)
        m1.metric("Study hours", f"{hours}")
        m2.metric("Chance of passing", f"{prob:.0%}")
        st.progress(float(prob))

        with st.expander("ℹ️ How was this prediction generated?", expanded=True):
            st.write(
                f"The app sent **{hours} study hours** to a Logistic Regression model "
                f"that was trained earlier and loaded from `student_pass_model.joblib`. "
                f"The model estimated a **{prob:.0%}** chance of passing, and since that is "
                f"{'at least' if pred == 1 else 'below'} 50%, it predicted "
                f"**{'Pass' if pred == 1 else 'Fail'}**."
            )

        st.subheader("📈 Study hours vs. predicted class")
        hours_range = [h / 2 for h in range(0, 25)]
        preds = model.predict(pd.DataFrame({"study_hours": hours_range}))
        chart_df = pd.DataFrame(
            {"Predicted class (1 = Pass, 0 = Fail)": preds},
            index=pd.Index(hours_range, name="Study hours"),
        )
        st.line_chart(chart_df)

st.caption("IT Elective 3 — Machine Learning | Model Persistence and Deployment")