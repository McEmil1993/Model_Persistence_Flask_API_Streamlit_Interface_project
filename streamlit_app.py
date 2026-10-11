
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


st.title("Student Pass/Fail Prediction")
st.write(
    "Enter the student's information to predict "
    "a result based on study hours."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Please run train_model.py first."
    )
    st.stop()

student_name = st.text_input("Student Name")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Result", type="primary"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        input_data = pd.DataFrame({
            "StudyHours": [study_hours]
        })

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")

        if prediction == "Pass":
            st.success("Predicted Result: Pass")
        else:
            st.warning("Predicted Result: Fail")

        st.info(
            "The application loaded the saved KNN model "
            "and provided the study hours as input. "
            "The model compares this input with patterns "
            "learned from the training data to generate "
            "a prediction."
        )

        chart_data = pd.DataFrame({
            "Category": ["Study Hours", "Predicted Class"],
            "Value": [
                study_hours,
                1 if prediction == "Pass" else 0
            ]
        })

        st.subheader("Study Hours and Predicted Class")
        st.bar_chart(chart_data.set_index("Category"))

        st.caption(
            "Predicted Class: 1 = Pass, 0 = Fail. "
            "The chart summarizes the input and prediction."
        )

