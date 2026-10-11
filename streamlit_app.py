
import streamlit as st
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction System Ate daisy")
st.write("Predict a student's result based on study hours.")

# Input fields
student_name = st.text_input("Enter Student Name")
study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

# Predict button
if st.button("Predict Result"):
    if student_name.strip() == "":
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]

        if prediction == 1:
            st.success(f"{student_name} is predicted to PASS!")
            st.write("Keep up the good work!")
        else:
            st.error(f"{student_name} is predicted to FAIL.")
            st.write("More study time may help improve the result.")