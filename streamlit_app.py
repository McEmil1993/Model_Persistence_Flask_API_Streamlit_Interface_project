
import streamlit as st
import joblib

st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓"
)

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

st.title("Student Pass Prediction")
st.write("Predict a student's result using study hours.")

student_name = st.text_input("Student Name")
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
        prediction = int(model.predict([[study_hours]])[0])
        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction Result")
        st.write(f"Student: {student_name}")
        st.write(f"Study Hours: {study_hours}")
        st.success(f"Predicted Result: {result}")
        st.info(
            "This is a classroom demonstration using a small sample dataset, "
            "not a reliable real-world assessment."
        )
