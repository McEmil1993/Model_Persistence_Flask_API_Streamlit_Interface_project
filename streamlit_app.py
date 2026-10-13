"""Streamlit interface for student pass predictions."""

import pandas as pd
import streamlit as st

from train_model import load_model


st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")
st.write("Enter a student's name and weekly study hours to predict the class.")

student_name = st.text_input("Student name")
study_hours = st.number_input("Study hours", min_value=0.0, step=0.5)

if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        model = load_model()
        prediction = int(model.predict([[study_hours]])[0])
        result = "Pass" if prediction == 1 else "Fail"
        st.success(f"Prediction for {student_name}: {result}")
        st.info(
            f"The saved machine-learning model used {study_hours:g} study hours as "
            "its input and classified the student as likely to pass or fail."
        )

        st.subheader("Prediction summary")
        chart_data = pd.DataFrame(
            {
                "Metric": ["Study hours", "Predicted class"],
                "Value": [study_hours, prediction],
            }
        )
        st.bar_chart(chart_data, x="Metric", y="Value", horizontal=True)
        st.caption("Predicted class: 0 = Fail, 1 = Pass")
