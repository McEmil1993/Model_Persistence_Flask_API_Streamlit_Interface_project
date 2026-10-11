import streamlit as st
import joblib

model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

student_name = st.text_input("Student Name")
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=3.0,
    step=0.5
)

if st.button("Predict Result"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]

        if prediction == "Pass":
            st.success(f"{student_name} is predicted to PASS!")
        else:
            st.error(f"{student_name} is predicted to FAIL.")

        st.write("### How was this prediction generated?")
        st.write(
            f"The trained Decision Tree model used "
            f"{study_hours} study hours to predict the result. "
            "The model was trained using sample study-hour data "
            "and their corresponding Pass/Fail results."
        )
