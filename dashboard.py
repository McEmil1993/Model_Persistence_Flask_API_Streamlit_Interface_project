import os
import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Student Academic Performance Prediction System",
    page_icon="🎓",
    layout="centered"
)

# SAFETY: Use environment variable for API URL, fallback to localhost for local dev
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000/predict")
REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT", 10))

# Header
st.title("🎓 Student Academic Performance Prediction System")
st.write(
    "This system utilizes a trained machine learning model to predict "
    "a student's academic outcome based on their study habits and attendance."
)

st.divider()

# Student input form
with st.form("prediction_form"):
    st.subheader("Student Information")

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5,
        help="Average number of hours the student studies daily."
    )

    attendance = st.number_input(
        "Attendance Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0,
        help="Student's class attendance rate."
    )

    submitted = st.form_submit_button(
        "Generate Prediction",
        use_container_width=True
    )

# Send inputs to Flask API
if submitted:
    payload = {
        "study_hours": study_hours,
        "attendance_percentage": attendance
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        if response.ok:
            # SAFETY: Catch JSON parsing errors
            try:
                result = response.json()
            except ValueError:
                st.error("The API returned an invalid JSON response.")
                st.stop()

            # SAFETY: Check required keys
            required_keys = {"prediction", "study_hours", "attendance_percentage"}
            if not required_keys.issubset(result.keys()):
                st.error(f"API response is missing required fields. Got: {list(result.keys())}")
                st.stop()

            st.subheader("Prediction Result")

            # SAFETY: Case-insensitive comparison
            prediction = str(result["prediction"]).strip().lower()
            if prediction == "pass":
                st.success("Predicted Outcome: PASS")
            else:
                st.error("Predicted Outcome: FAIL")

            col1, col2 = st.columns(2)

            col1.metric(
                "Study Hours",
                f'{result["study_hours"]:.1f} hours'
            )

            col2.metric(
                "Attendance",
                f'{result["attendance_percentage"]:.1f}%'
            )

            st.info(result.get("note", ""))

        else:
            try:
                error_message = response.json().get(
                    "error", "The API returned an error."
                )
            except ValueError:
                error_message = f"HTTP {response.status_code}: The API returned a non-JSON response."

            st.error(f"Prediction failed: {error_message}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the Flask API. "
            "Please ensure the backend service is running on port 5000."
        )

    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Please try again.")

    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")

    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")

st.divider()

st.caption(
    "Note: Predictions are based on study hours only. "
    "Attendance does not currently affect the trained model."
)