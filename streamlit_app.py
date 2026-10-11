import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Pass/Fail Prediction System")
st.write("Machine Learning Deployment - By: Rezamae Quinlog ")

# 1. Load the trained model (Checks both paths so it never errors)
model_path = "student_pass_model.joblib"
if not os.path.exists(model_path):
    model_path = "student_prediction_project/student_pass_model.joblib"

try:
    model = joblib.load(model_path)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model not found! Please run 'python train_model.py' first. Error: {e}")

st.divider()

# ==========================================
# REQUIRED INPUTS (Exercise 3 Tasks)
# ==========================================

# Requirement 1: A student name input
student_name = st.text_input("👤 Student Name:", value="Juan Dela Cruz", placeholder="Enter student name here...")

# Requirement 2: A study-hours input
study_hours = st.number_input(
    "⏱️ Daily Study Hours:",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
    help="Number of hours spent studying each day"
)

# Attendance input (for model accuracy)
attendance = st.slider(
    "📅 Attendance Percentage (%):",
    min_value=0,
    max_value=100,
    value=75,
    help="Student classroom attendance rate"
)

st.write("")

# Requirement 3: A prediction button
predict_button = st.button("🚀 Predict Outcome", type="primary", use_container_width=True)

# ==========================================
# PREDICTION & RESULT DISPLAY
# ==========================================
if predict_button:
    if not student_name.strip():
        st.warning("⚠️ Please enter a student name.")
    elif not model_loaded:
        st.error("Model is not loaded. Train the model first.")
    else:
        # Perform model inference
        input_data = np.array([[study_hours, attendance]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.divider()

        # Requirement 4: A result message
        if prediction == 1:
            st.success(f"### 🎉 Congratulations, {student_name}!")
            st.success(f"**Predicted Class: PASS** (Confidence: {probability:.1%})")
            st.balloons()
        else:
            st.error(f"### ⚠️ Warning, {student_name}!")
            st.error(f"**Predicted Class: FAIL** (Pass Probability: {probability:.1%})")

        # Requirement 5: A short explanation of how the prediction was generated
        with st.expander("ℹ️ How was this prediction generated?", expanded=True):
            st.markdown(f"""
            **Explanation:**
            - **Input Features:** {study_hours} study hours/day and {attendance}% attendance.
            - **Algorithm:** The prediction was generated using a **Logistic Regression** classifier loaded via `joblib`.
            - **Decision Logic:** The model computes a linear combination:
              $$\\text{{Score}} = (w_1 \\times \\text{{Study Hours}}) + (w_2 \\times \\text{{Attendance}}) + b$$
            - The score is passed through the **Sigmoid function** to produce a probability between 0% and 100%.
            - Since the calculated probability is **{probability:.1%}** (threshold is 50%), the predicted class is **{"PASS" if prediction == 1 else "FAIL"}**.
            """)

        # Requirement 6 (BONUS): Add a chart that compares the study hours with the predicted class
        st.divider()
        st.subheader("📊 Comparison Chart: Study Hours vs. Predicted Class")
        
        # Historical reference data
        chart_data = pd.DataFrame({
            'Study Hours': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0],
            'Outcome': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 0: Fail, 1: Pass
        })

        fig, ax = plt.subplots(figsize=(7, 3.5))

        # Plot historical students (Fail = Red, Pass = Green) 
        fail_data = chart_data[chart_data['Outcome'] == 0]
        pass_data = chart_data[chart_data['Outcome'] == 1]
        
        ax.scatter(fail_data['Study Hours'], fail_data['Outcome'], color='red', label='Class: Fail (0)', s=60, alpha=0.6)
        ax.scatter(pass_data['Study Hours'], pass_data['Outcome'], color='green', label='Class: Pass (1)', s=60, alpha=0.6)

        # Highlight current student (Blue Star)
        ax.scatter([study_hours], [prediction], color='blue', s=220, marker='*', label=f'Current: {student_name}', zorder=5)

        # Draw decision boundary threshold line at 4.0 hours
        ax.axvline(x=4.0, color='gray', linestyle='--', linewidth=1.5, label='Pass Threshold (4.0 hrs)')

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Fail (0)', 'Pass (1)'])
        ax.set_xlabel('Daily Study Hours')
        ax.set_ylabel('Predicted Class')
        ax.set_title('Study Hours vs. Student Outcome (Pass / Fail)')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='center left')

        st.pyplot(fig)
        st.caption("★ The blue star indicates your input compared to the historical pass/fail distribution.")