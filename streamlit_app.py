import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Pass Prediction System")
st.write("Predict student performance based on study hours.")

if os.path.exists("student_pass_model.joblib"):
    model = joblib.load("student_pass_model.joblib")

    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    if st.button("Predict Result"):
        try:
            prediction = model.predict(
                [[study_hours, attendance]]
            )[0]

            if prediction == 1:
              st.success("Result: PASS 🎉")
    
            else:
             st.error("Result: FAIL")

        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.error(
        "Model file not found. Please check "
        "student_pass_model.joblib."
    )
