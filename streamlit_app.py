import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction")
st.write("Predict a student's result based on study hours.")

# Student inputs
student_name = st.text_input("Enter Student Name")

study_hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=1.0
)

attendance_percentage = st.number_input(
    "Enter Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=1.0
)

# Prediction button
if st.button("Predict Result"):
    if student_name.strip() == "":
        st.warning("Please enter the student's name.")
    else:
        # Prepare model input
        student_data = pd.DataFrame({
            "StudyHours": [study_hours]
        })

        # Generate prediction
        prediction = model.predict(student_data)[0]

        # Display result
        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")
        st.write(f"Attendance Percentage: {attendance_percentage}%")

        if prediction == "Pass":
            st.success(
                f"Congratulations, {student_name}! "
                "Predicted Result: Pass"
            )
        else:
            st.error(
                f"Predicted Result: Fail. "
                f"Keep studying, {student_name}!"
            )

        # Explain the prediction
        st.subheader("How was the prediction generated?")
        st.write(
            f"The trained Decision Tree model used {study_hours} "
            f"study hours as input and predicted {prediction}. "
            "Attendance is displayed for reference but is not "
            "currently used by the trained model. The model was "
            "loaded from a saved file, so it did not need to be "
            "trained again."
        )

        # Bonus chart
        st.subheader("Bonus: Study Hours vs Predicted Class")

        hours_range = list(range(1, 11))

        chart_data = pd.DataFrame({
            "Study Hours": hours_range
        })

        chart_data["Prediction"] = model.predict(
            pd.DataFrame({"StudyHours": hours_range})
        )

        chart_data["ClassValue"] = chart_data["Prediction"].map({
            "Fail": 0,
            "Pass": 1
        })

        st.bar_chart(
            chart_data,
            x="Study Hours",
            y="ClassValue"
        )

        st.caption(
            "0 = Fail, 1 = Pass. The chart shows predictions "
            "for 1 to 10 study hours. Predictions do not "
            "guarantee actual student results."
        )