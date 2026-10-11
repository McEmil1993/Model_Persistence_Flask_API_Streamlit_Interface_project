
import streamlit as st
import joblib
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
    value=5.0,
    step=0.5
)

# Prediction button
if st.button("Predict Result"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = int(model.predict([[hours]])[0])
        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction Result")
        st.write(f"Student Name: {name}")
        st.write(f"Study Hours: {hours}")

        if prediction == 1:
            st.success(f"Predicted Result: {result}")
        else:
            st.error(f"Predicted Result: {result}")

        st.write(
            "The prediction was generated using the study hours "
            "entered and patterns learned by the trained model. "
            "This is an estimate, not a guarantee of actual grades."
        )

        # Bonus chart
        fig, ax = plt.subplots()
        ax.bar(
            ["Study Hours", "Predicted Class"],
            [hours, prediction]
        )
        ax.set_ylabel("Value")
        ax.set_title("Study Hours vs Predicted Class")
        st.pyplot(fig)
        plt.close(fig)
