import os

import requests
import streamlit as st


st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="📘",
    layout="centered",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:5000/predict")
REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT", "10"))

st.title("Student Pass Prediction")
st.write("Enter a student's details to get a prediction from the trained model.")

with st.form("prediction_form"):
    st.subheader("Student Information")
    student_name = st.text_input("Student name")
    study_hours = st.number_input(
        "Study hours per day", min_value=0.0, max_value=24.0, value=5.0, step=0.5
    )
    attendance = st.number_input(
        "Attendance percentage", min_value=0.0, max_value=100.0, value=90.0, step=1.0
    )
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    if not student_name.strip():
        st.error("Please enter the student's name.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={
                    "study_hours": study_hours,
                    "attendance_percentage": attendance,
                },
                timeout=REQUEST_TIMEOUT,
            )
            try:
                result = response.json()
            except ValueError:
                st.error("The API returned an invalid response.")
            else:
                if not response.ok:
                    st.error(result.get("error", f"Request failed ({response.status_code})."))
                elif "prediction" not in result:
                    st.error("The API response did not include a prediction.")
                else:
                    prediction = str(result["prediction"]).strip().lower()
                    st.subheader(f"Prediction for {student_name.strip()}")
                    if prediction == "pass":
                        st.success("Predicted outcome: PASS")
                    else:
                        st.error("Predicted outcome: FAIL")
                    st.write(
                        "The saved decision tree model generated this prediction "
                        "from the student's study hours. Attendance is validated "
                        "and returned by the API, but the current model was trained "
                        "using study hours only, so attendance does not affect the result."
                    )
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the Flask API. Start app.py and try again.")
        except requests.exceptions.Timeout:
            st.error("The Flask API took too long to respond.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")

st.caption("The current model uses study hours only; attendance does not affect its prediction.")
