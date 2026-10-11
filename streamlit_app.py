import streamlit as st
import pandas as pd
from joblib import load

model = load('student_pass_model.joblib')

st.title("Student Performance Prediction System")
st.write("This application helps predict whether a student will pass or fail based on study hours and attendance percentage.")

student_name = st.text_input("Enter Student Name:")
study_hours = st.number_input("Enter Study Hours:", min_value=0.0, max_value=24.0, step=0.5)
attendance_percentage = st.number_input("Enter Attendance Percentage:", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict"):
    if student_name.strip() == "":
        st.warning("Please enter a student name.")
    else:
        if attendance_percentage < 75.0 or study_hours <= 2.0:
            prediction = "Fail"
        else:
            prediction = model.predict([[study_hours, attendance_percentage]])[0]

        if prediction == "Pass":
            st.success(f"Result for {student_name}: The predicted outcome is **{prediction}**.")
        else:
            st.error(f"Result for {student_name}: The predicted outcome is **{prediction}**.")

        st.info(f"Explanation: Evaluated {study_hours} study hours and {attendance_percentage}% attendance against academic thresholds.")

        st.markdown("""
        ### Why doesn't the model need to be trained again?
        The model does not need to be trained again because model persistence saves the trained model's learned weights and parameters into the `student_pass_model.joblib` file. 
        We can simply load this file into memory and instantly generate predictions for new inputs without repeating the training process.
        """)

        st.write("### Input Metrics Comparison Chart")
        chart_data = pd.DataFrame({
            'Metrics': ['Study Hours', 'Attendance (%)'],
            'Values': [study_hours, attendance_percentage]
        }).set_index('Metrics')
        
        st.bar_chart(chart_data)