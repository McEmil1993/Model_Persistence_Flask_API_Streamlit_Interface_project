
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="centered"
)

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"

# Page title and introduction
st.title("🎓 Student Pass/Fail Predictor")
st.write(
    "Enter the student's information below to predict "
    "whether the student is likely to pass or fail."
)

st.divider()

# Student inputs
name = st.text_input(
    "Student Name",
    placeholder="Enter student name"
)

hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.slider(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=75
)

# Prediction button
if st.button("Predict Result", type="primary"):

    if not name.strip():
        st.warning("Please enter the student's name.")

    else:
        try:
            # Send student data to Flask API
            response = requests.post(
                API_URL,
                json={
                    "study_hours": hours,
                    "attendance_percentage": attendance
                },
                timeout=10
            )

            # Check API response
            if response.status_code == 200:
                result = response.json()

                prediction = result["prediction"]
                probability = result["pass_probability"]

                # Display prediction result
                st.subheader(f"Result for {name.strip()}")

                if prediction == "Pass":
                    st.success("Prediction: PASS")
                else:
                    st.error("Prediction: FAIL")

                # Display pass probability
                st.metric(
                    "Estimated Pass Probability",
                    f"{probability * 100:.2f}%"
                )

                # Display submitted information
                st.write(f"**Student Name:** {name.strip()}")
                st.write(f"**Study Hours:** {hours}")
                st.write(f"**Attendance:** {attendance}%")

                # Explain the prediction
                st.subheader("How Was the Prediction Generated?")

                st.write(
                    "The application sent the student's study hours "
                    "and attendance percentage to the Flask API. "
                    "The API used the saved Logistic Regression model "
                    "to predict Pass or Fail."
                )

                st.info(
                    "The model was trained using sample student data. "
                    "Its prediction is an estimate, not a guarantee "
                    "of actual academic performance."
                )

                # Bonus chart
                st.subheader("Study Hours vs. Predicted Class")

                chart_hours = list(range(1, 10))
                chart_results = []

                # Predict for each study-hour value
                for h in chart_hours:
                    chart_response = requests.post(
                        API_URL,
                        json={
                            "study_hours": h,
                            "attendance_percentage": attendance
                        },
                        timeout=10
                    )

                    chart_response.raise_for_status()

                    chart_prediction = (
                        chart_response.json()["prediction"]
                    )

                    chart_results.append(
                        1 if chart_prediction == "Pass" else 0
                    )

                # Create chart
                chart_data = pd.DataFrame({
                    "Study Hours": chart_hours,
                    "Predicted Class": chart_results
                })

                fig, ax = plt.subplots(figsize=(8, 4))

                ax.plot(
                    chart_data["Study Hours"],
                    chart_data["Predicted Class"],
                    marker="o"
                )

                ax.set_xlabel("Study Hours")
                ax.set_ylabel("Predicted Class")
                ax.set_yticks(
                    [0, 1],
                    ["Fail", "Pass"]
                )
                ax.set_xticks(chart_hours)
                ax.set_title(
                    f"Predictions at {attendance}% Attendance"
                )
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
                plt.close(fig)

            else:
                try:
                    error_message = response.json().get(
                        "error",
                        "The prediction request failed."
                    )
                except ValueError:
                    error_message = "The prediction request failed."

                st.error(error_message)

        except requests.RequestException:
            st.error(
                "Cannot connect to the Flask API. "
                "Please make sure app.py is running."
            )