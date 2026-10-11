
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/student_pass_model.joblib")

st.title("Student Performance Predictor")

st.write(
    "Enter the student's name, study hours, and "
    "attendance percentage to predict performance."
)

# Student information
student_name = st.text_input("Student Name")

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance Percentage (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=1.0
)

# Prediction button
if st.button("Predict Performance"):

    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        # Predict using BOTH features
        prediction = model.predict([
            [study_hours, attendance]
        ])[0]

        result = "Pass" if prediction == 1 else "Fail"

        # Display result
        st.subheader("Prediction Result")

        st.write(f"**Student:** {student_name}")
        st.write(f"**Study Hours:** {study_hours}")
        st.write(f"**Attendance:** {attendance}%")
        st.write(f"**Predicted Result:** {result}")

        if result == "Pass":
            st.success("The model predicts Pass.")
        else:
            st.error("The model predicts Fail.")

        st.info(
            "This prediction uses both study hours and "
            "attendance percentage. It is a classroom "
            "demonstration, not a guarantee of actual "
            "academic performance."
        )

        # Chart showing predictions for different
        # attendance percentages at the entered study hours
        st.subheader("Attendance vs. Predicted Class")

        attendance_values = list(range(0, 101, 10))

        predictions = model.predict([
            [study_hours, value]
            for value in attendance_values
        ])

        chart_data = pd.DataFrame({
            "Attendance (%)": attendance_values,
            "Predicted Class": [
                "Pass" if value == 1 else "Fail"
                for value in predictions
            ]
        })

        chart_data["Predicted Class Number"] = (
            chart_data["Predicted Class"]
            .map({"Fail": 0, "Pass": 1})
        )

        st.line_chart(
            chart_data.set_index("Attendance (%)")[
                "Predicted Class Number"
            ]
        )

        st.caption(
            "Chart values: Fail = 0, Pass = 1. "
            "Study hours remain fixed while attendance "
            "changes. The chart shows model predictions, "
            "not actual student results."
        )
