
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("student_pass_model.joblib")

# App title
st.title("Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

# Student information
student_name = st.text_input("Enter Student Name")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Predict button
if st.button("Predict"):
    if student_name.strip() == "":
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")

        if prediction == "Pass":
            st.success("Prediction: PASS")
        else:
            st.error("Prediction: FAIL")

        st.info(
            "Explanation: This prediction is based on the "
            "student's study hours using a trained Decision Tree model."
        )

        # Bonus: Chart showing study hours and predicted result
        chart_data = pd.DataFrame({
            "Study Hours": [study_hours],
            "Predicted Result": [1 if prediction == "Pass" else 0]
        })

        st.subheader("Study Hours and Predicted Result")
        st.bar_chart(
            chart_data.set_index("Study Hours")["Predicted Result"]
        )

        st.caption("Chart value: 1 = Pass, 0 = Fail.")
