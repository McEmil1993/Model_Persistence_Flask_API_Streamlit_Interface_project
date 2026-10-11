import streamlit as st
import joblib
import os
import pandas as pd

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")
st.write("Enter student details to get an example prediction.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "student_pass_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

student_name = st.text_input("Student name", placeholder="Enter student name")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=24.0,
    value=5.0, step=0.5
)

if st.button("Predict result"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = int(model.predict([[study_hours]])[0])
        label = "Likely to Pass" if prediction == 1 else "Needs Improvement"

        st.subheader("Prediction result")
        st.success(f"{student_name.strip()}: {label}")
        st.write(
            f"The model generated this result using the entered study time "
            f"({study_hours:g} hours) and patterns in its example training data. "
            "This is a demonstration, not a guarantee of academic performance."
        )

        chart_data = pd.DataFrame({
            "Study hours": [study_hours],
            "Predicted class (0 = Needs Improvement, 1 = Likely to Pass)": [prediction]
        })
        st.bar_chart(chart_data.set_index("Study hours"))

st.caption("Demo only: this model predicts using study hours only.")
