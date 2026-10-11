from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("A simple machine-learning demonstration based on study hours.")

if not MODEL_PATH.exists():
    st.error("Model file not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

student_name = st.text_input("Student name", placeholder="Enter student name")
study_hours = st.number_input(
    "Study hours per day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
)

if st.button("Predict", type="primary"):
    if not student_name.strip():
        st.warning("Please enter a student name.")
    else:
        predicted_class = int(model.predict([[float(study_hours)]])[0])
        prediction = "Pass" if predicted_class == 1 else "Fail"

        st.subheader("Prediction result")
        st.write(f"**Student:** {student_name.strip()}")
        st.write(f"**Study hours:** {study_hours:g} hours per day")

        if prediction == "Pass":
            st.success(f"Predicted class: {prediction}")
        else:
            st.warning(f"Predicted class: {prediction}")

        st.caption(
            "Explanation: The application sends the entered study hours "
            "to the saved, previously trained model. The model returns a "
            "Pass/Fail class based on patterns in the demonstration training data. "
            "This is only a classroom demonstration, not a reliable assessment "
            "of a real student's performance."
        )

        chart_data = pd.DataFrame(
            {
                "Measure": ["Study hours", "Predicted class (Fail=0, Pass=1)"],
                "Value": [float(study_hours), predicted_class],
            }
        )
        st.subheader("Bonus: input and predicted class")
        st.bar_chart(
            chart_data,
            x="Measure",
            y="Value",
            x_label="Measure",
            y_label="Hours or class code",
            sort=False,
            alt="Study hours and predicted class, with Fail encoded as 0 and Pass as 1",
        )
