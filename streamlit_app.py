
import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load the saved model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "student_pass_model.joblib"
)

model = joblib.load(MODEL_PATH)

# Application title
st.title("🎓 Student Pass Prediction System")
st.write("Predict a student's academic result using study hours.")

st.divider()

# Student information
st.subheader("Student Information")

student_name = st.text_input("Enter Student Name")

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
        # Generate prediction using the saved model
        prediction = model.predict([[study_hours]])[0]

        st.divider()
        st.subheader("Prediction Result")

        st.write(f"**Student Name:** {student_name}")
        st.write(f"**Study Hours:** {study_hours} hours")

        if prediction == "Pass":
            st.success("Predicted Result: PASS")
        else:
            st.error("Predicted Result: FAIL")

        # Explanation
        st.subheader("How was the prediction generated?")

        st.info(
            "The system loads the previously trained machine learning "
            "model from student_pass_model.joblib. It uses the student's "
            "study hours to predict the result. The model does not use "
            "attendance percentage in its current prediction."
        )

        # Bonus chart
        st.subheader("Study Hours and Predicted Class")

        class_value = 1 if prediction == "Pass" else 0

        fig, ax = plt.subplots()

        bars = ax.bar(
            ["Study Hours", "Predicted Class"],
            [study_hours, class_value]
        )

        ax.set_ylabel("Value")
        ax.set_title("Student Prediction Chart")
        ax.set_ylim(0, max(10, study_hours + 2))

        ax.bar_label(bars)
        ax.text(
            1,
            class_value + 0.3,
            prediction,
            ha="center"
        )

        st.pyplot(fig)
        plt.close(fig)

st.divider()
st.caption("Student Pass Prediction System | Machine Learning Project")
