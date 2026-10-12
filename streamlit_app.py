import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Configure the application
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Application title
st.title("Student Performance Prediction")

st.write(
    "Enter a student's name and daily study hours "
    "to predict the student's result."
)

# Input fields
student_name = st.text_input("Student Name")

study_hours = st.number_input(
    "Study Hours per Day",
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
        prediction = model.predict([[study_hours]])[0]

        # Display results
        st.subheader("Prediction Result")

        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours:g}")

        if prediction == "Pass":
            st.success(f"Predicted Result: {prediction}")
        else:
            st.error(f"Predicted Result: {prediction}")

        # Explain the prediction
        st.info(
            "This prediction uses a trained Logistic Regression model "
            "loaded from a saved joblib file. The model uses example "
            "training data and predicts based on study hours only."
        )

        # Create a simple chart
        st.subheader("Study Hours and Predicted Result")

        fig, ax = plt.subplots()

        ax.bar(
            ["Study Hours", "Predicted Class"],
            [
                study_hours,
                1 if prediction == "Pass" else 0
            ]
        )

        ax.set_ylabel("Value")
        ax.set_title("Student Prediction Overview")

        st.pyplot(fig)

        plt.close(fig)