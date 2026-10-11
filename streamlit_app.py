
import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction")
st.write("Predict a student's result based on study hours.")

# Student inputs
name = st.text_input("Student Name")

hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Prediction button
if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        # Prepare input in the format expected by the model
        features = pd.DataFrame({
            "study_hours": [hours]
        })

        # Make prediction
        prediction = model.predict(features)[0]
        result = "Pass" if prediction == 1 else "Fail"

        # Display result
        st.subheader("Prediction Result")
        st.write(f"Student Name: {name}")
        st.write(f"Study Hours: {hours}")

        if prediction == 1:
            st.success(f"Predicted Result: {result}")
        else:
            st.error(f"Predicted Result: {result}")

        # Explain the prediction
        st.subheader("How was this prediction generated?")
        st.write(
            "The saved machine learning model used the student's "
            "study hours as input. It compared the input with patterns "
            "learned from the training data to predict Pass or Fail. "
            "The saved model was loaded instead of being trained again."
        )

        # Bonus chart
        st.subheader("Study Hours vs Predicted Class")

        chart_data = pd.DataFrame({
            "Category": ["Study Hours", "Predicted Class (0=Fail, 1=Pass)"],
            "Value": [hours, int(prediction)]
        })

        st.bar_chart(chart_data.set_index("Category"))
