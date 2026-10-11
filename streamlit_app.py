
import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.title("Student Performance Prediction")

# Load saved model
model = joblib.load("student_pass_model.joblib")

# Student inputs
name = st.text_input("Student Name")
hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=2.0
)

# Prediction button
if st.button("Predict Result"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[hours]])[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {name}")
        st.success(f"Predicted Result: {prediction}")

        st.write(
            "The prediction was generated using the "
            "student's study hours and the saved trained model."
        )

        # Bonus chart
        fig, ax = plt.subplots()
        ax.bar(["Study Hours"], [hours])
        ax.set_ylabel("Hours")
        ax.set_title("Student Study Hours")
        st.pyplot(fig)
