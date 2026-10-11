
import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run train_model.py first.")
    st.stop()

st.title("🎓 Student Pass Prediction")
st.write("Enter the student's information below.")

# Student information
student_name = st.text_input("Student Name")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance Percentage (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=1.0
)

if st.button("Predict Result"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        # Current model predicts using study hours only
        prediction = int(model.predict([[study_hours]])[0])

        st.write(f"**Student:** {student_name}")
        st.write(f"**Study Hours:** {study_hours}")
        st.write(f"**Attendance:** {attendance}%")

        if prediction == 1:
            st.success("Prediction: Pass")
        else:
            st.error("Prediction: Fail")

        st.info(
            "Note: The current model uses study hours only. "
            "Attendance is recorded but does not affect this prediction. "
            "The result is based on sample training data."
        )

st.subheader("Study Hours and Sample Training Data")

hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

fig, ax = plt.subplots()
ax.scatter(hours, results)
ax.set_xlabel("Study Hours")
ax.set_ylabel("Sample Result")
ax.set_yticks([0, 1], ["Fail", "Pass"])
ax.set_title("Sample Training Data")
st.pyplot(fig)
