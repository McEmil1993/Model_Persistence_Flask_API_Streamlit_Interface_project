
import streamlit as st
import joblib
import pandas as pd

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

# Prediction button
if st.button("Predict Result"):

    if student_name.strip() == "":
        st.warning("Please enter the student's name.")

    else:
        # Predict student result
        prediction = model.predict([[study_hours]])[0]

        if prediction == 1:
            result = "Pass"
            st.success(f"{student_name} is predicted to Pass.")
        else:
            result = "Fail"
            st.error(f"{student_name} is predicted to Fail.")

        # Prediction details
        st.subheader("Prediction Details")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")
        st.write(f"Predicted Class: {result}")

        st.write(
            "Explanation: This sample model uses study hours "
            "to estimate the result."
        )

        # Comparison chart only
        st.subheader("Study Hours vs. Predicted Class")

        hours_range = [[hours] for hours in range(1, 9)]
        predictions = model.predict(hours_range)

        chart_data = pd.DataFrame({
            "Study Hours": list(range(1, 9)),
            "Predicted Class": predictions
        })

        st.line_chart(
            chart_data.set_index("Study Hours")
        )

        st.caption(
            "Predicted Class: 0 = Fail, 1 = Pass. "
            "The chart compares study hours from 1 to 8 "
            "with the model's predicted results."
        )
