"""A simple Streamlit page for predicting a student's chance of passing."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Set the browser tab title and the main page heading.
st.set_page_config(page_title="Student Pass Predictor")
st.title("Student Pass Predictor")
st.write("Enter a student's information, then ask the saved model for a prediction.")

# Load the trained model once when Streamlit starts this app.
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(MODEL_PATH)

# These are the same fictional training examples used in train_model.py.
# They are included here only to calculate average study hours for the chart.
training_data = pd.DataFrame(
    {
        "study_hours": [
            0, 1, 1, 2, 2, 3, 3, 4, 4, 5,
            5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
        ],
        "passed": [
            0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ],
    }
)

# Collect the values the user wants to predict.
student_name = st.text_input("Student name")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=10.0, value=5.0, step=0.5
)
attendance = st.number_input(
    "Attendance percentage", min_value=0.0, max_value=100.0, value=75.0, step=1.0
)

# Run a prediction only after the user clicks the button.
if st.button("Predict"):
    # Give an unnamed student a friendly label rather than showing a blank name.
    display_name = student_name.strip() or "This student"

    # The feature order must match the order used when the model was trained.
    features = [[study_hours, attendance]]
    prediction = model.predict(features)[0]
    pass_probability = model.predict_proba(features)[0][1]
    result = "Pass" if prediction == 1 else "Fail"

    # Show the final label in a clear sentence.
    st.success(f"{display_name} is predicted to: {result}")

    # Explain both the inputs and the probability returned by the model.
    st.write(
        "The model used the entered study hours and attendance percentage "
        "to estimate the probability of passing."
    )
    st.write(f"Estimated probability of passing: **{pass_probability:.1%}**")

    # Compare the entered study hours with the two training-data averages.
    average_hours = training_data.groupby("passed")["study_hours"].mean()
    chart_data = pd.DataFrame(
        {
            "Study hours": [
                study_hours,
                average_hours[1],
                average_hours[0],
            ]
        },
        index=["Entered student", "Average: passed", "Average: failed"],
    )
    st.subheader("Study-hours comparison")
    st.bar_chart(chart_data)
