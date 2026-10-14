import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction System")

# Student Name
student_name = st.text_input("Student Name")

# Study Hours
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    step=0.5
)

# Predict Button
if st.button("Predict"):

    prediction = model.predict([[study_hours]])[0]

    st.subheader("Prediction Result")

    if prediction == "Pass":
        st.success(
            f"{student_name}, the model predicts that you will PASS."
        )
    else:
        st.error(
            f"{student_name}, the model predicts that you will FAIL."
        )

    st.write("### How was this prediction generated?")
    st.write(
        """
        The Machine Learning model analyzed the number of
        study hours entered. Based on patterns learned
        during training, it predicted whether the student
        is more likely to pass or fail.
        """
    )

    # Bonus Chart
    chart_data = pd.DataFrame({
        "Study Hours": [study_hours],
        "Predicted Class": [1 if prediction == "Pass" else 0]
    })

    st.write("### Study Hours vs Predicted Class")
    st.bar_chart(chart_data.set_index("Study Hours"))