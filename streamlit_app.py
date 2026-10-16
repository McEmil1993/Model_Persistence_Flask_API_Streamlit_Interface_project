
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Page configuration
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

# App title
st.title("🎓 Student Pass Prediction System")
st.write("Predict whether a student will pass based on study hours.")

# Student information
student_name = st.text_input("Student Name")

study_hours = st.number_input(
    "Study Hours per Day",
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
        # Make prediction
        prediction = model.predict([[study_hours]])[0]

        result = "PASS" if prediction == 1 else "FAIL"

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours} hours per day")

        if prediction == 1:
            st.success(f"Congratulations, {student_name}! Prediction: PASS 🎉")
        else:
            st.error(f"{student_name}, prediction: FAIL")

        st.info(
            "This prediction is based on study hours only. "
            "It is a machine learning estimate, not a guaranteed result."
        )

        # Bonus: chart showing study hours and predicted class
        st.subheader("📊 Prediction Chart")

        predicted_class = int(prediction)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].bar(["Study Hours"], [study_hours])
        axes[0].set_title("Study Hours per Day")
        axes[0].set_ylabel("Hours")
        axes[0].set_ylim(0, 24)

        axes[1].bar(
            ["Predicted Class"],
            [predicted_class],
        )
        axes[1].set_title("Predicted Class")
        axes[1].set_ylabel("Class (0 = FAIL, 1 = PASS)")
        axes[1].set_ylim(0, 1.2)
        axes[1].set_yticks([0, 1], ["FAIL (0)", "PASS (1)"])

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
