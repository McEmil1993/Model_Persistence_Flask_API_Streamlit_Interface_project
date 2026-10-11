
import streamlit as st
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

st.title("Student Performance Predictor")
st.write("Predict a student's result based on study hours.")

# Student information
student_name = st.text_input("Enter student name")

study_hours = st.number_input(
    "Enter study hours",
    min_value=0.0,
    max_value=24.0,
    value=2.0,
    step=0.5
)

attendance_percentage = st.number_input(
    "Enter Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=1.0
)
# Make a prediction when the button is clicked
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]

        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction Result")
        st.write(f"Student name: {student_name}")
        st.write(f"Study hours: {study_hours}")
        st.write(f"Predicted result: {result}")

        st.info(
            "This prediction is based on study hours "
            "using a previously trained machine-learning model."
        )
