
import streamlit as st
import matplotlib.pyplot as plt
import requests

API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="📚",
    layout="centered"
)

st.title("Student Pass Prediction")
st.write("Enter the student's information below.")

# Student input fields
student_name = st.text_input(
    "Student Name",
    placeholder="Enter student name"
)

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=1.0
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=1.0
)

if st.button("Predict Result", type="primary"):
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
            response.raise_for_status()
            result = response.json()

            st.subheader("Student Prediction")
            st.write("**Student Name:**", student_name)
            st.write("**Study Hours:**", study_hours)

            prediction = str(result["prediction"]).strip()

            if prediction.lower() == "pass":
                st.success(f"Prediction: {prediction}")
            else:
                st.error(f"Prediction: {prediction}")

            st.caption(result.get("note", ""))

        except requests.RequestException:
            st.error(
                "Cannot connect to the API. "
                "Please run `python app.py` in another terminal."
            )

# Prediction graph
st.subheader("Study Hours vs. Prediction")

hours = list(range(25))
predictions = []

try:
    for hour in hours:
        response = requests.post(
            API_URL,
            json={
                "study_hours": hour,
                "attendance_percentage": attendance
            },
            timeout=5
        )
        response.raise_for_status()
        result = response.json()

        predictions.append(
            1 if str(result["prediction"]).strip().lower() == "pass"
            else 0
        )

    current_response = requests.post(
        API_URL,
        json={
            "study_hours": study_hours,
            "attendance_percentage": attendance
        },
        timeout=5
    )
    current_response.raise_for_status()
    current_result = current_response.json()

    current_y = (
        1 if str(current_result["prediction"]).strip().lower() == "pass"
        else 0
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.step(
        hours,
        predictions,
        where="post",
        color="blue",
        linewidth=2,
        label="Model Prediction"
    )

    ax.scatter(
        [study_hours],
        [current_y],
        color="red",
        s=100,
        label="Selected Study Hours",
        zorder=5
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Prediction")
    ax.set_xticks(range(0, 25, 2))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.15, 1.15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

except requests.RequestException:
    st.info("Start the Flask API to display the graph.")
