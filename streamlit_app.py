
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Pass Prediction System")
st.write("Predict a student's exam result based on study hours.")

# Load model
MODEL_FILE = Path(__file__).parent / "student_pass_model.joblib"

if not MODEL_FILE.exists():
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

st.divider()
st.subheader("Student Information")

# Input form
with st.form("student_form"):
    student_name = st.text_input("Student Name")

    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    submitted = st.form_submit_button(
        "Predict Result",
        use_container_width=True
    )

# Store prediction history for the current session
if "results" not in st.session_state:
    st.session_state.results = []

if submitted:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        input_data = pd.DataFrame({
            "study_hours": [study_hours]
        })

        prediction = int(model.predict(input_data)[0])
        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        prediction_class = "Pass" if prediction == 1 else "Fail"

        st.session_state.results.append({
            "Student Name": student_name.strip(),
            "Study Hours": study_hours,
            "Prediction Class": prediction_class
        })

        st.divider()
        st.subheader("Prediction Result")

        st.write(f"**Student Name:** {student_name}")
        st.write(f"**Study Hours:** {study_hours}")
        
        if prediction == 1:
            st.success("Prediction Class: PASS 🎉")
        else:
            st.error("Prediction Class: FAIL")

        st.metric("Estimated Pass Probability", f"{probability:.2%}")

# Comparison chart
if st.session_state.results:
    st.divider()
    st.subheader("Study Hours vs. Prediction Class")

    results_df = pd.DataFrame(st.session_state.results)

    st.dataframe(results_df, use_container_width=True)

    # Convert prediction classes to numbers for charting
    chart_df = results_df.copy()
    chart_df["Prediction Value"] = chart_df["Prediction Class"].map({
        "Fail": 0,
        "Pass": 1
    })

    st.caption("Prediction Class: 0 = Fail | 1 = Pass")

    st.scatter_chart(
        chart_df,
        x="Study Hours",
        y="Prediction Value",
        color="Prediction Class",
        size=100
    )

    st.subheader("Student Comparison")
    st.bar_chart(
        chart_df.set_index("Student Name")[["Study Hours"]]
    )

    if st.button("Clear Comparison History"):
        st.session_state.results = []
        st.rerun()

st.divider()
st.caption("Student Pass Prediction System | Machine Learning Project")
