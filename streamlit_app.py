
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Pass Prediction", page_icon="🎓")

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

st.title("Student Pass Prediction")
st.write("Predict whether a student is likely to pass.")

name = st.text_input("Student Name")
study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)
attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=1.0
)

if st.button("Predict Result"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        features = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance_percentage": attendance
        }])

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        st.subheader(f"Result for {name}")

        if prediction == 1:
            st.success("Prediction: PASS")
        else:
            st.error("Prediction: FAIL")

        st.write(f"Probability of passing: {probability:.1%}")
        st.caption("This prediction uses a sample-trained model.")
