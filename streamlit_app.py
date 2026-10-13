
import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Student Prediction System",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Prediction System")
st.write(
    "Predict a student's academic outcome using "
    "study hours and attendance."
)

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"

# Store previous predictions for the chart
if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Student Information")

with st.form("prediction_form"):
    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student name"
    )

    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0
    )

    predict_button = st.form_submit_button(
        "🔍 Predict Result"
    )

if predict_button:
    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        payload = {
            "study_hours": study_hours,
            "attendance_percentage": attendance
        }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=10
            )

            response.raise_for_status()
            result = response.json()

            prediction = result["prediction"]
            probability = result.get("pass_probability")

            # Save prediction for the chart
            st.session_state.history.append({
                "student_name": student_name.strip(),
                "study_hours": study_hours,
                "attendance": attendance,
                "predicted_class": str(prediction),
                "class_code": (
                    1 if str(prediction).lower() == "pass"
                    else 0
                )
            })

            # Display prediction results
            st.subheader("Prediction Result")

            st.write(f"**Student:** {student_name}")
            st.write(f"**Study Hours:** {study_hours}")
            st.write(f"**Attendance:** {attendance}%")

            if str(prediction).lower() == "pass":
                st.success("Prediction: PASS 🎉")
            else:
                st.error("Prediction: FAIL")

            if probability is not None:
                st.metric(
                    "Pass Probability",
                    f"{float(probability) * 100:.1f}%"
                )

            # Explain how prediction works
            st.subheader(
                "How Was the Prediction Generated?"
            )

            st.write(
                "The system sends the student's study hours "
                "and attendance percentage to the Flask API. "
                "The trained machine-learning model processes "
                "these inputs and returns the predicted class "
                "and pass probability."
            )

            st.info(
                "This prediction is an estimate from the "
                "trained model, not a guarantee of the "
                "student's actual academic result."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the Flask API. "
                "Please run 'python app.py' in another "
                "terminal first."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction API took too long to respond. "
                "Please try again."
            )

        except requests.exceptions.RequestException as error:
            st.error(f"API request failed: {error}")

        except (ValueError, KeyError, TypeError):
            st.error(
                "The API returned an unexpected response. "
                "Please check your Flask prediction code."
            )

# Bonus chart
st.divider()
st.subheader(
    "📊 Bonus: Study Hours vs Predicted Class"
)

if st.session_state.history:
    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.caption(
        "Each point represents one prediction. "
        "Pass = 1 and Fail = 0 on the vertical axis."
    )

    st.scatter_chart(
        history_df,
        x="study_hours",
        y="class_code",
        color="predicted_class",
        height=350
    )

    st.subheader("Prediction History")

    st.dataframe(
        history_df[
            [
                "student_name",
                "study_hours",
                "attendance",
                "predicted_class"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

else:
    st.info(
        "Make a prediction to display the chart "
        "and prediction history."
    )
