
import streamlit as st
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass/Fail Prediction")
st.write("Enter student information to get a sample prediction.")

# Student inputs
student_name = st.text_input("Student Name")
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Result"):
    if student_name.strip() == "":
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]

        if prediction == 1:
            st.success(f"{student_name} is predicted to Pass.")
        else:
            st.error(f"{student_name} is predicted to Fail.")

        st.write(
            "Explanation: This sample model uses study hours "
            "to estimate the result."
        )