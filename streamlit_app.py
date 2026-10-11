
import streamlit as st
import joblib

# Page settings
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the model. Check your model filename.")
    st.write(e)
    st.stop()

# App title
st.title("🎓 Student Performance Prediction")
st.write("Enter the student's information to predict the result.")

# Student inputs
student_name = st.text_input("Student Name")

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=2.0,
    step=0.5
)

# Prediction button
if st.button("Predict Result", type="primary", use_container_width=True):

    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        try:
            prediction = model.predict([[study_hours]])[0]
            result = str(prediction).strip().lower()

            st.subheader("Prediction Result")
            st.write(f"**Student Name:** {student_name}")
            st.write(f"**Study Hours:** {study_hours}")

            # Handle text labels
            if result in ["pass", "passed"]:
                st.success("✅ PASS")
                st.write("The model predicts that the student will pass.")

            elif result in ["fail", "failed"]:
                st.error("❌ FAIL")
                st.write("The model predicts that the student will fail.")

            # Handle numeric labels: 1 = Pass, 0 = Fail
            elif result in ["1", "1.0"]:
                st.success("✅ PASS")
                st.write("The model predicts that the student will pass.")

            elif result in ["0", "0.0"]:
                st.error("❌ FAIL")
                st.write("The model predicts that the student will fail.")

            else:
                st.warning("The model returned a label that is not recognized.")
                st.write(f"Raw model prediction: {prediction}")
                st.info(
                    "Check the labels used when training your model. "
                    "The code assumes 1 = Pass and 0 = Fail for numeric labels."
                )

            st.caption(
                "This result is a machine learning prediction, "
                "not a guarantee of the student's actual outcome."
            )

        except Exception as e:
            st.error("Prediction failed. Please check your model and input.")
            st.write(e)
