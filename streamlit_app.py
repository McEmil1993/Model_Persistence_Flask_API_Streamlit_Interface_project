import streamlit as st
import pandas as pd
import joblib
import os

st.title("🎓 Student Performance Prediction Dashboard")
st.write("Enter student details below to predict whether they will pass or fail.")

# Load Model Safely
MODEL_PATH = 'student_pass_model.joblib'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model file not found! Please run `train_model.py` first.")
    model = None

# User Inputs
student_name = st.text_input("Student Name", placeholder="e.g., Shaine Anana")
study_hours = st.number_input("Study Hours (per week)", min_value=0.0, max_value=40.0, value=4.0, step=0.5)
attendance = st.slider("Attendance Percentage (%)", min_value=0, max_value=100, value=75)

# Prediction Button & Results
if st.button("Predict Outcome"):
    if model is not None:
        # Make prediction
        features = [[study_hours, attendance]]
        prediction = model.predict(features)[0]
        
        # Result Message Formatting
        name_display = student_name if student_name.strip() else "The student"
        if prediction == 1:
            st.success(f"🎉 **{name_display}** is predicted to **PASS**!")
        else:
            st.error(f"⚠️ **{name_display}** is predicted to **FAIL**. Additional study hours and class attendance are recommended.")
        
        # Explanation of how the prediction was generated
        st.info(
            f"**How this prediction was generated:** "
            f"The trained machine learning model evaluated your inputs "
            f"(**{study_hours} study hours** and **{attendance}% attendance**) "
            f"against the historical parameters loaded from the persisted model file."
        )
        
        # Bonus: Comparison Chart
        st.subheader("📊 Input Metrics Comparison")
        chart_data = pd.DataFrame({
            "Metric": ["Study Hours", "Attendance (%) / 10"],
            "Value": [study_hours, attendance / 10]
        })
        st.bar_chart(chart_data.set_index("Metric"))
    else:
        st.warning("Cannot generate prediction because the model is missing.")