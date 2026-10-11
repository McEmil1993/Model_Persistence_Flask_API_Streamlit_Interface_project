
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load("student_pass_model.joblib")

st.title("Student Performance Prediction System")
st.write("Predict student performance based on study hours.")

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
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]
        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.success(f"Predicted Class: {result}")

        st.write(
            f"The model predicted the result using "
            f"{study_hours} study hours. "
            f"The saved model was loaded without retraining."
        )

        # Bonus chart
        predicted_class = 1 if prediction == 1 else 0

        fig, ax = plt.subplots()
        ax.bar(
            ["Study Hours", "Predicted Class"],
            [study_hours, predicted_class]
        )
        ax.set_ylabel("Value")
        ax.set_title("Study Hours vs Predicted Class")

        st.pyplot(fig)
        plt.close(fig)
