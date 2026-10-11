
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("student_pass_model.joblib")

st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

# Student information
student_name = st.text_input("Enter Student Name")

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Result"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")

        if prediction == "Pass":
            st.success("Prediction: PASS 🎉")
        else:
            st.error("Prediction: FAIL")

        st.info(
            "Explanation: This prediction is based on the "
            "study hours used by the trained model. It is an "
            "educational estimate, not a guaranteed result."
        )

        # Bonus chart
        st.subheader("Study Hours and Predicted Result")

        result_value = 1 if prediction == "Pass" else 0

        fig, ax = plt.subplots()
        ax.bar(
            ["Study Hours", "Prediction (1=Pass, 0=Fail)"],
            [study_hours, result_value]
        )
        ax.set_ylabel("Value")
        ax.set_title("Student Prediction Chart")

        st.pyplot(fig)
        plt.close(fig)
