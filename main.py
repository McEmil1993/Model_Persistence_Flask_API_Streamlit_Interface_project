
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

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
    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        # Generate prediction
        prediction = int(model.predict([[study_hours]])[0])

        result = "Pass" if prediction == 1 else "Fail"

        # Display result
        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")

        if prediction == 1:
            st.success(f"Predicted Result: {result}")
        else:
            st.error(f"Predicted Result: {result}")

        # Explain the prediction
        st.subheader("How was the prediction generated?")
        st.write(
            "The system uses a trained Logistic Regression model. "
            "It receives the student's study hours and predicts "
            "whether the student belongs to the Pass or Fail class. "
            "This is a model prediction, not a guaranteed actual result."
        )

        # Bonus: Chart comparing study hours and predicted class
        chart_data = pd.DataFrame({
            "Study Hours": [study_hours],
            "Predicted Class (0 = Fail, 1 = Pass)": [prediction]
        })

        st.subheader("Prediction Chart")
        st.bar_chart(
            chart_data,
            x="Study Hours",
            y="Predicted Class (0 = Fail, 1 = Pass)"
        )
