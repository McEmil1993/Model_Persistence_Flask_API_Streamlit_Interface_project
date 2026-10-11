import streamlit as st
import joblib

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

st.title("🎓 Student Pass Predictor")
st.write("Enter the student's details to predict Pass or Fail.")

name = st.text_input("Student Name", placeholder="e.g., Juan Dela Cruz")
study_hours = st.number_input("Study Hours per Week", 0.0, 50.0, 5.0, step=0.5)
attendance = st.slider("Attendance (%)", 0, 100, 75)

if st.button("Predict", use_container_width=True):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = model.predict([[study_hours, attendance]])[0]
        message = f"{name.strip()}: **{'Pass' if pred == 1 else 'Fail'}**"
        if pred == 1:
            st.success(message)
        else:
            st.error(message)
        st.caption(f"{study_hours:g} study hours/week · {attendance}% attendance")