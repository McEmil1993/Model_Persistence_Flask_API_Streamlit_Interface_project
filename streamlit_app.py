
import streamlit as st
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓"
)

# Page title and description
st.title("🎓 Student Performance Predictor")
st.write("Predict a student's performance based on study hours.")

# Student information
student_name = st.text_input("Enter Student Name")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Prediction button
if st.button("Predict Performance"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        # Make a prediction using the saved model
        prediction = model.predict([[study_hours]])[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")

        if prediction == 1:
            st.success(f"{student_name} is predicted to Pass.")
        else:
            st.error(f"{student_name} is predicted to Fail.")

        st.info(
            "This prediction is based on study hours "
            "and the sample training data. It is not a "
            "guarantee of actual academic performance."
            
        )
        
        # Explain how the prediction was generated
        st.subheader("💡 How Was This Prediction Generated?")

        st.write(
            "The system uses a trained Decision Tree Classifier "
            "to predict whether the student will pass or fail "
            "based on the number of study hours entered. "
            "The model learned from sample training data "
            "containing study hours and their corresponding "
            "pass or fail results. It then uses the entered "
            "study hours to generate a prediction."
        )

        # Bonus: Compare study hours with the predicted class
        import pandas as pd

        st.subheader("📊 Study Hours vs. Predicted Class")

        predicted_class = "Pass" if prediction == 1 else "Fail"

        chart_data = pd.DataFrame({
            "Metric": ["Study Hours", "Predicted Class (0 = Fail, 1 = Pass)"],
            "Value": [study_hours, prediction]
        })

        st.bar_chart(
            chart_data,
            x="Metric",
            y="Value"
        )

        st.write(f"Predicted Class: {predicted_class}")

        st.caption(
            "The chart displays the entered study hours and the model's "
            "numeric class prediction. Class 0 means Fail, and class 1 means Pass."
        )
