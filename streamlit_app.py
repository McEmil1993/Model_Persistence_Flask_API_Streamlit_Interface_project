"""
Student Pass Prediction - Streamlit Interface
Exercise 3: Student name, study hours, attendance percentage, prediction, explanation, and chart.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(page_title="Student Pass Prediction", page_icon="🎓")
st.title("🎓 Student Pass Prediction")
st.write("Enter a student's information to predict the class based on study hours.")

if not MODEL_PATH.exists():
    st.error("The trained model was not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

student_name = st.text_input("Student name", placeholder="Enter student name")
study_hours = st.number_input(
    "Study hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
)
attendance_percentage = st.number_input(
    "Attendance percentage (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=1.0,
)

if st.button("Predict Result", type="primary"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        # The current model uses study_hours only; attendance is displayed but
        # does not influence the prediction unless the model is retrained with it.
        prediction = int(model.predict([[study_hours]])[0])
        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction Result")
        st.write(f"**Student:** {student_name.strip()}")
        st.write(f"**Study hours:** {study_hours:g}")
        st.write(f"**Attendance percentage:** {attendance_percentage:g}%")
        if prediction == 1:
            st.success(f"Predicted class: {result}")
        else:
            st.error(f"Predicted class: {result}")

        st.info(
            f"The model used {study_hours:g} study hours as its input and returned "
            f"class {prediction} ({result}). Attendance was recorded as "
            f"{attendance_percentage:g}%, but it does not affect this prediction "
            "because the current model was trained using study hours only. "
            "The trained model was loaded from student_pass_model.joblib and was "
            "not retrained for this prediction."
        )

        # Bonus chart: compare study hours and predicted class.
        fig, ax = plt.subplots()
        ax.bar(
            ["Study hours", "Predicted class (0=Fail, 1=Pass)"],
            [study_hours, prediction],
        )
        ax.set_title(f"Study Hours and Predicted Class: {student_name.strip()}")
        ax.set_ylabel("Value")
        ax.set_ylim(bottom=0)
        st.pyplot(fig)
        plt.close(fig)

st.caption(
    "Demonstration model trained on example data. Attendance is displayed but "
    "does not affect predictions unless the model is retrained using attendance data. "
    "Replace the example dataset with your class dataset if one was provided."
)
