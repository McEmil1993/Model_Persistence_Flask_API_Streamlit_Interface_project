"""
Exercise 3 - Streamlit Interface
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Prediction System")
st.write("Enter the details below and click **Predict**.")

model = joblib.load("student_pass_model.joblib")

# Inputs
student_name = st.text_input("Student Name", placeholder="e.g. Juan Dela Cruz")
study_hours  = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

# Button
if st.button("🔮 Predict"):
    if not student_name.strip():
        st.warning("Please enter the student name.")
    else:
        pred = int(model.predict(np.array([[study_hours]]))[0])
        label = "Pass ✅" if pred == 1 else "Fail ❌"

        if pred == 1:
            st.success(f"**{student_name}** is predicted to: **{label}**")
        else:
            st.error(f"**{student_name}** is predicted to: **{label}**")

        st.info(
            "**How the prediction was generated:**\n\n"
            "The saved Logistic Regression model (`student_pass_model.joblib`) "
            "takes the study hours as input, computes a weighted sum with its "
            "learned coefficients, and applies a sigmoid function. If the "
            "probability is ≥ 0.5, the result is **Pass**; otherwise **Fail**."
        )

        # Bonus chart
        st.subheader("📊 Study Hours vs Predicted Class")
        df = pd.DataFrame({
            "Study Hours": [study_hours],
            "Prediction":  [pred]
        })
        st.bar_chart(df.set_index("Study Hours"))
