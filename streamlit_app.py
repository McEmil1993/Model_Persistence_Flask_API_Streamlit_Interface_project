import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Student ML Predictor", page_icon="📈")

st.title("📊 Advanced Student Outcome Predictor")
st.write("Provide the student's metrics below to evaluate academic performance likelihood.")

# Load Model Safely
MODEL_PATH = 'student_pass_model.joblib'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model file not found! Please run `train_model.py` first.")
    model = None

# User Inputs
student_name = st.text_input("Student Name", placeholder="e.g., Aubry Miles Postanes")
study_hours = st.slider("Weekly Study Hours", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
prev_grade = st.number_input("Previous Grade Average (%)", min_value=0.0, max_value=100.0, value=78.0, step=1.0)
attendance = st.slider("Attendance Rate (%)", min_value=0, max_value=100, value=80)

# Prediction Button & Results
if st.button("Run Prediction Analysis"):
    if model is not None:
        # Make prediction
        features = [[study_hours, prev_grade, attendance]]
        prediction = model.predict(features)[0]
        
        # Result Message Formatting
        name_display = student_name if student_name.strip() else "The student"
        if prediction == 1:
            st.success(f"🎉 **{name_display}** is evaluated to **PASS** based on current indicators!")
        else:
            st.error(f"⚠️ **{name_display}** is evaluated to **FAIL**. Intervention, tutoring, or higher attendance is recommended.")
        
        # Explanation
        st.info(
            f"**Prediction breakdown:** The Random Forest model evaluated study habits "
            f"({study_hours} hrs), past academic standing ({prev_grade}%), and presence "
            f"({attendance}%) against pattern trees learned during training."
        )
        
        # Visual Comparison Chart
        st.subheader("📈 Student Metrics Overview")
        chart_df = pd.DataFrame({
            "Indicator": ["Study Hours (Scaled)", "Previous Grade", "Attendance"],
            "Score": [study_hours * 5, prev_grade, attendance]  # Scaled study hours for better visual balance
        })
        st.bar_chart(chart_df.set_index("Indicator"))
    else:
        st.warning("Prediction unavailable due to missing model file.")