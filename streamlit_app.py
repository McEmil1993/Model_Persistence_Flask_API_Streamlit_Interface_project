import streamlit as st
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Student Pass/Fail Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load the trained model
MODEL_PATH = Path(__file__).parent / "student_pass_model.joblib"

try:
    predictor = joblib.load(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

# App title
st.title("🎓 Student Pass/Fail Prediction")
st.write("Enter the student's information to predict the result.")

st.divider()

# Student information
name = st.text_input("Student Name", placeholder="Enter student name")

hours_spent = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Prediction
if st.button("Predict Result", type="primary"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        try:
            outcome = predictor.predict([[hours_spent]])[0]

            if outcome == 1:
                st.success(f"{name} is predicted to Pass! 🎉")
            else:
                st.error(f"{name} is predicted to Fail.")

            st.write(
                f"Study hours entered: {hours_spent} hours"
            )

            st.caption(
                "This prediction is based on the trained "
                "machine learning model using study hours."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")