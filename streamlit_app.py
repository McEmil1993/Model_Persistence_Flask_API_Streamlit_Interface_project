
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

# Page title
st.title("🎓 Student Pass Prediction")
st.write(
    "Predict a student's result based on their study hours."
)

st.divider()

# Student inputs
student_name = st.text_input(
    "Student Name",
    placeholder="Enter student name"
)

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Prediction button
if st.button("Predict Result", type="primary"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        # Predict using the saved model
        prediction = int(model.predict([[study_hours]])[0])

        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction Result")
        st.write(f"**Student:** {student_name}")
        st.write(f"**Study Hours:** {study_hours}")

        if prediction == 1:
            st.success(f"{student_name} is predicted to Pass!")
        else:
            st.error(f"{student_name} is predicted to Fail.")

        # Explain the prediction
        st.subheader("How was the prediction generated?")
        st.write(
            "The system loaded a previously trained "
            "K-Nearest Neighbors (KNN) model. It compared "
            "the entered study hours with the training data "
            "and predicted the class using nearby examples. "
            "The model was not retrained when you clicked "
            "the prediction button."
        )

        # Bonus: Chart comparing hours and predicted class
        st.subheader("Study Hours vs. Predicted Class")

        class_value = 1 if prediction == 1 else 0

        chart_data = pd.DataFrame({
            "Category": ["Study Hours", "Predicted Class"],
            "Value": [study_hours, class_value]
        })

        fig, ax = plt.subplots()
        bars = ax.bar(
            chart_data["Category"],
            chart_data["Value"]
        )

        ax.set_title("Student Prediction Summary")
        ax.set_ylabel("Value")
        ax.set_ylim(0, max(10, study_hours + 1))
        ax.bar_label(bars, fmt="%.1f")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Predicted Class: 1 = Pass, 0 = Fail. "
            "This chart compares study hours with the "
            "numeric class label; the two values use "
            "different scales of meaning."
        )

st.divider()
st.caption("Student Pass Prediction System | Machine Learning")
