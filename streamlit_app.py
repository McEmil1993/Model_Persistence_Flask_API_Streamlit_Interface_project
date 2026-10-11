import streamlit as st
import requests
import pandas as pd

# Page settings
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title and description
st.title("🎓 Student Pass Prediction System")

st.write(
    "Enter the student's information to predict "
    "whether the student will pass or fail."
)

st.divider()

# Student inputs
student_name = st.text_input("Student Name")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=1.0
)

# Prediction button
if st.button("Predict", type="primary"):

    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        try:
            with st.spinner("Generating prediction..."):

                response = requests.post(
                    "http://127.0.0.1:5000/predict",
                    json={
                        "study_hours": study_hours,
                        "attendance_percentage": attendance
                    },
                    timeout=10
                )

            if response.ok:
                result = response.json()
                prediction = result.get("prediction")

                if prediction is None:
                    st.error(
                        "The API response has no 'prediction' field."
                    )

                else:
                    st.subheader("Prediction Result")
                    st.write(f"**Student Name:** {student_name}")
                    st.write(f"**Study Hours:** {study_hours}")
                    st.write(f"**Attendance:** {attendance}%")

                    if str(prediction).strip().lower() == "pass":
                        st.success(f"Prediction: {prediction}")
                    elif str(prediction).strip().lower() == "fail":
                        st.error(f"Prediction: {prediction}")
                    else:
                        st.info(f"Prediction: {prediction}")

                    st.subheader("Study Hours Chart")

                    chart_data = pd.DataFrame(
                        {"Study Hours": [study_hours]},
                        index=[student_name]
                    )

                    st.bar_chart(chart_data)

                    st.subheader(
                        "How was the prediction generated?"
                    )

                    st.write(
                        "The application sends the study hours and "
                        "attendance percentage to the Flask API. "
                        "The API uses the saved Machine Learning "
                        "model to generate the prediction and "
                        "returns the result as JSON."
                    )

                    st.info(
                        "Note: If the trained model uses study hours "
                        "only, attendance will not affect the "
                        "prediction until the model is retrained "
                        "using attendance data."
                    )

            else:
                st.error(
                    f"API Error ({response.status_code}): "
                    f"{response.text}"
                )

        except requests.exceptions.RequestException as error:
            st.error(
                "Could not connect to the Flask API. "
                "Make sure app.py is running on port 5000."
            )
            st.caption(str(error))