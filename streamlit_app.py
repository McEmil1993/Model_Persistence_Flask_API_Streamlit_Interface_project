"""
Exercise 3 - Streamlit Interface
Run with:  streamlit run streamlit_app.py
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")   # saved in Exercise 1

model = load_model()

st.title("🎓 Student Pass/Fail Predictor")
st.write("Enter a student's study hours to predict whether they will pass.")

# ---- Inputs ----
name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

# ---- Prediction button ----
if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        sample = pd.DataFrame({"study_hours": [hours]})
        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]

        # ---- Result message ----
        if pred == 1:
            st.success(f"✅ {name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"❌ {name} is predicted to FAIL (pass probability {prob:.0%}).")

        # ---- Short explanation ----
        st.info(
            f"**How this was generated:** the saved model (student_pass_model.joblib) "
            f"learned from past examples of study hours and results. It took "
            f"{hours} study hours as input and calculated a {prob:.0%} chance of passing. "
            f"If that chance is 50% or higher, it predicts Pass; otherwise Fail. "
            f"The model was loaded from file, not retrained."
        )

        # ---- Bonus chart: study hours vs. predicted class ----
        st.subheader("Study hours vs. predicted class")
        hrs = [x / 2 for x in range(0, 25)]  # 0 to 12 hours
        curve = pd.DataFrame({
            "Study hours": hrs,
            "Predicted class (0=Fail, 1=Pass)": model.predict(pd.DataFrame({"study_hours": hrs})),
        }).set_index("Study hours")
        st.line_chart(curve)
        st.caption(f"{name}'s input ({hours} hours) is predicted as {'Pass (1)' if pred == 1 else 'Fail (0)'}.")
