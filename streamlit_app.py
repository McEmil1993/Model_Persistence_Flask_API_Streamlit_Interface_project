
# Streamlit Student Pass Prediction Interface
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓"
)

st.title("Student Pass Prediction")
st.write(
    "Enter a student's name, study hours, and attendance "
    "to predict the result."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run 'python train_model.py' first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

student_name = st.text_input("Student name")

study_hours = st.number_input(
    "Study hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.5
)

attendance_percentage = st.slider(
    "Attendance percentage (%)",
    min_value=0,
    max_value=100,
    value=80,
    step=1
)

if st.button("Predict Result"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        # Use both inputs in the prediction.
        input_values = [[
            float(study_hours),
            float(attendance_percentage)
        ]]

        prediction_value = int(
            model.predict(input_values)[0]
        )

        result = "Pass" if prediction_value == 1 else "Fail"

        st.subheader("Prediction Result")

        if result == "Pass":
            st.success(
                f"{student_name.strip()} is predicted to Pass."
            )
        else:
            st.error(
                f"{student_name.strip()} is predicted to Fail."
            )

        st.write(
            f"**Study hours:** {study_hours:g}"
        )

        st.write(
            f"**Attendance:** {attendance_percentage}%"
        )

        st.write(
            "**Explanation:** The saved machine-learning model "
            "uses both study hours and attendance to generate "
            "the prediction."
        )

        st.caption(
            "This demonstration uses illustrative sample data, "
            "not real student records. Its predictions are not "
            "validated assessments of actual academic performance."
        )

        # Chart 1: Compare study hours while keeping attendance fixed.
        hours_to_compare = list(range(0, 25))

        hour_inputs = [
            [hours, float(attendance_percentage)]
            for hours in hours_to_compare
        ]

        hour_predictions = model.predict(hour_inputs)

        hours_chart = pd.DataFrame({
            "Study hours": hours_to_compare,
            "Predicted class": [
                int(value) for value in hour_predictions
            ]
        })

        st.subheader("Study Hours vs. Predicted Class")

        st.caption(
            f"Attendance is held constant at "
            f"{attendance_percentage}%. "
            "0 = Fail, 1 = Pass."
        )

        st.line_chart(
            hours_chart.set_index("Study hours")
        )

        # Chart 2: Compare attendance while keeping study hours fixed.
        attendance_values = list(range(0, 101, 5))

        attendance_inputs = [
            [float(study_hours), attendance]
            for attendance in attendance_values
        ]

        attendance_predictions = model.predict(
            attendance_inputs
        )

        attendance_chart = pd.DataFrame({
            "Attendance percentage": attendance_values,
            "Predicted class": [
                int(value) for value in attendance_predictions
            ]
        })

        st.subheader("Attendance vs. Predicted Class")

        st.caption(
            f"Study hours are held constant at {study_hours:g}. "
            "0 = Fail, 1 = Pass."
        )

        st.line_chart(
            attendance_chart.set_index("Attendance percentage")
        )
