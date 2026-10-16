import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Page title
st.title("Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

# Student name input
student_name = st.text_input("Enter student name")

# Study hours input
study_hours = st.number_input(
    "Enter study hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Prediction button
if st.button("Predict Result"):

    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        # Make a prediction
        prediction = int(model.predict([[study_hours]])[0])

        result = "Pass" if prediction == 1 else "Fail"

        # Display the result
        st.subheader("Prediction Result")
        st.write("Student:", student_name)
        st.write("Study hours:", study_hours)

        if prediction == 1:
            st.success(f"{student_name} is predicted to Pass!")
        else:
            st.error(f"{student_name} is predicted to Fail.")

        # Explain how the prediction was generated
        st.info(
            "This prediction was generated using a trained "
            "Logistic Regression model. The model uses study "
            "hours as its input. Attendance is not included "
            "in the current model."
        )

        # Bonus: Display a chart
        chart_data = pd.DataFrame({
            "Predicted Class": [result],
            "Study Hours": [study_hours]
        })

        fig, ax = plt.subplots()

        color = "green" if prediction == 1 else "red"

        ax.bar(
            chart_data["Predicted Class"],
            chart_data["Study Hours"],
            color=color
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Study Hours")
        ax.set_title("Study Hours vs Predicted Class")

        st.pyplot(fig)