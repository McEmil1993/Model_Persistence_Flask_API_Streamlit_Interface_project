
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Prediction")

st.title("Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

# Student name
name = st.text_input("Student Name")

# Study hours
hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Predict button
if st.button("Predict Result"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = int(model.predict([[hours]])[0])
        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction Result")
        st.write(f"Student Name: {name}")
        st.write(f"Study Hours: {hours}")

        if prediction == 1:
            st.success(f"{name} is predicted to PASS!")
        else:
            st.error(f"{name} is predicted to FAIL.")

        st.info(
            "The prediction was generated using a trained "
            "machine learning model. It evaluates the study "
            "hours entered based on patterns learned from "
            "training data. This is an estimate, not a guarantee."
        )

        # Bonus chart: compare study hours and predicted class
        chart_data = pd.DataFrame({
            "Study Hours": [hours],
            "Predicted Class (0=Fail, 1=Pass)": [prediction]
        })

        st.subheader("Study Hours vs. Predicted Class")
        st.bar_chart(
            chart_data,
            x="Study Hours",
            y="Predicted Class (0=Fail, 1=Pass)"
        )

