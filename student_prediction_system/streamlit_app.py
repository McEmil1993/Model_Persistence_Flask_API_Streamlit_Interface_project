import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:5001/predict"

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")
st.write("Enter student details to request a prediction from the Flask backend.")

with st.form("prediction_form"):
    student_name = st.text_input("Student name")
    study_hours = st.number_input(
        "Study hours", min_value=0.0, max_value=24.0,
        value=5.0, step=0.5
    )
    attendance = st.number_input(
        "Attendance percentage (%)", min_value=0.0, max_value=100.0,
        value=90.0, step=1.0
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={
                    "study_hours": study_hours,
                    "attendance_percentage": attendance
                },
                timeout=5
            )
            if not response.ok:
                st.error(response.json().get("error", "API request failed."))
            else:
                result = response.json()
                st.subheader(f"Result for {student_name.strip()}")
                if result["prediction"] == "Pass":
                    st.success(f"Predicted class: {result['prediction']}")
                else:
                    st.warning(f"Predicted class: {result['prediction']}")

                st.write(f"**Study hours:** {result['study_hours']}")
                st.write(
                    f"**Attendance:** {result['attendance_percentage']}%"
                )
                st.info(
                    "How was this generated? The frontend sent the entered "
                    "values to Flask. Flask loaded the saved joblib model and "
                    "predicted using study hours. Attendance is shown but "
                    "does not influence this model yet."
                )

                chart_data = pd.DataFrame({
                    "Measure": ["Study hours", "Attendance (%)"],
                    "Entered value": [study_hours, attendance]
                }).set_index("Measure")
                st.bar_chart(chart_data)
        except requests.RequestException:
            st.error(
                "Cannot connect to Flask. Start the backend first with "
                "`python app.py`, then retry."
            )
