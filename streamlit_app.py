
import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

# Student inputs
student_name = st.text_input("Enter Student Name")

study_hours = st.number_input(
    "Enter Study Hours",
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
        prediction = int(model.predict([[study_hours]])[0])
        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")

        # Display prediction result
        if prediction == 1:
            st.success(f"Predicted Result: {result}")
        else:
            st.error(f"Predicted Result: {result}")

        # Explain the prediction
        st.write(
            "Explanation: The trained machine learning model "
            "used the student's study hours to predict the result. "
            "The prediction is based on patterns learned from "
            "the training data and is not a guarantee of actual performance."
        )

        # Bonus: Chart comparing study hours with predicted class
        chart_data = pd.DataFrame({
            "Study Hours": [study_hours],
            "Predicted Class (0 = Fail, 1 = Pass)": [prediction]
        })

        st.subheader("Study Hours and Predicted Class")

        st.bar_chart(
            chart_data,
            x="Study Hours",
            y="Predicted Class (0 = Fail, 1 = Pass)"
        )
