
import streamlit as st
import joblib
import pandas as pd

# Load the saved machine learning model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction")
st.write(
    "Enter the student's name and study hours "
    "to predict the result."
)

# Student information
student_name = st.text_input("Student Name")
study_hours = st.number_input(
    "Study Hours Per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Predict when the button is clicked
if st.button("Predict Result"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")

        if prediction == "Pass":
            st.success(f"{student_name} is predicted to Pass.")
        else:
            st.error(f"{student_name} is predicted to Fail.")

        st.info(
            "This is a demonstration prediction based on "
            "study hours and a small sample training dataset. "
            "It is not a guarantee of actual academic performance."
        )
                # Bonus: Chart comparing study hours and predicted class
        hours_list = list(range(1, 9))

        predictions = [
            model.predict([[hours]])[0]
            for hours in hours_list
        ]

        chart_data = pd.DataFrame({
            "Study Hours": hours_list,
            "Predicted Class": [
                1 if result == "Pass" else 0
                for result in predictions
            ]
        })

        st.subheader("Study Hours vs. Predicted Class")
        st.caption("1 = Pass, 0 = Fail")

        st.bar_chart(
            chart_data,
            x="Study Hours",
            y="Predicted Class"
        )
