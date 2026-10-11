
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

# Student information
name = st.text_input("Student Name")

hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=6.0,
    step=0.5
)

# Predict button
if st.button("Predict Result"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        # Make prediction
        prediction = int(model.predict([[hours]])[0])

        if prediction == 1:
            result = "Pass"
            st.success(f"{name} is predicted to Pass!")
        else:
            result = "Fail"
            st.error(f"{name} is predicted to Fail.")

        # Display prediction details
        st.subheader("Prediction Details")
        st.write("Student Name:", name)
        st.write("Study Hours:", hours)
        st.write("Predicted Result:", result)

        # Explain prediction
        st.info(
            "The prediction was generated using a previously "
            "trained machine learning model. The model uses "
            "patterns learned from study hours in the training data."
        )

        # Bonus: Chart comparing study hours and predicted class
        st.subheader("Study Hours vs. Predicted Class")

        chart_data = pd.DataFrame({
            "Category": ["Study Hours", "Predicted Class"],
            "Value": [hours, prediction]
        })

        fig, ax = plt.subplots()
        bars = ax.bar(
            chart_data["Category"],
            chart_data["Value"]
        )

        ax.set_ylabel("Value")
        ax.set_title("Study Hours vs. Predicted Class")
        ax.set_ylim(0, max(10, hours + 2))

        # Display values on top of the bars
        ax.bar_label(bars, fmt="%.1f")

        st.pyplot(fig)

        st.caption(
            "Predicted Class: 0 = Fail, 1 = Pass. "
            "The chart compares study hours with the encoded class."
        )
