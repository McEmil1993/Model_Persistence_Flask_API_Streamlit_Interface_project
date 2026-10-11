
import streamlit as st
import joblib

# Load the saved model
predictor = joblib.load("student_pass_model.joblib")

st.title("Student Pass/Fail Prediction")
st.write("Enter student information to get a sample prediction.")

# Student inputs
name = st.text_input("Student Name")

hours_spent = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Result"):
    if name.strip() == "":
        st.warning("Please enter the student's name.")
    else:
        outcome = predictor.predict([[hours_spent]])[0]

        if outcome == 1:
            st.success(f"{name} is predicted to Pass.")
        else:
            st.error(f"{name} is predicted to Fail.")

        st.write(
            "Explanation: This sample model uses study hours "
            "to estimate the result."
        )