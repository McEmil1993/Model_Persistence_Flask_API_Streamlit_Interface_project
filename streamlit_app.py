# streamlit_app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass/Fail Predictor")

# Inputs
name = st.text_input("Student Name")
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
attendance = st.slider("Attendance Percentage", 0, 100, 75)

if st.button("Predict"):
    # Prediction
    pred = model.predict(np.array([[study_hours, attendance]]))[0]
    result = "Pass" if pred == 1 else "Fail"

    # Result message
    st.success(f"Student: **{name}**")
    st.write(f"Prediction: **{result}**")

    # Short explanation
    st.info("""
**How was this prediction made?**  
The model was trained using study hours and attendance.  
Higher study hours and attendance increase the chance of getting a **Pass**.
""")

    chart_data = pd.DataFrame({
        "Feature": ["Study Hours", "Attendance (%)"],
        "Value": [study_hours, attendance]
    })

    fig = px.bar(
        chart_data,
        x="Feature",
        y="Value",
        color="Feature",
        color_discrete_sequence=["#1f77b4", "#ff7f0e"],
        text="Value",
        title="Student Input Overview"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis_title="Value",
        xaxis_title="",
        title_x=0.5,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)