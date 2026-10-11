import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(
    page_title="Student Pass Predictor", 
    page_icon="🎓", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS / FULL GRAPHICS & STYLING
# ==========================================
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        letter-spacing: 0.5px;
    }

    /* Card Containers */
    .css-1r6slb0, .stTextInput, .stNumberInput, .stSlider {
        background-color: white;
    }
    
    .card-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.8);
    }

    /* Custom Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #5a6fd6 0%, #684192 100%);
    }

    /* Success & Error Box Overrides */
    .stSuccess {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%) !important;
        border-radius: 12px;
        padding: 1rem;
        border: none;
    }
    .stError {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%) !important;
        border-radius: 12px;
        padding: 1rem;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown("""
    <div class="main-header">
        <h1>🎓 Student Pass/Fail Prediction System</h1>
        <p>Advanced Machine Learning Deployment &bull; Developed by: <b>cyrich</b></p>
    </div>
""", unsafe_allow_html=True)

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

# ==========================================
# REQUIRED INPUTS CONTAINER
# ==========================================
st.markdown('<div class="card-container">', unsafe_allow_html=True)
st.subheader("📝 Enter Student Details")

# Requirement 1: A student name input
student_name = st.text_input("👤 Student Name:", value="Juan Dela Cruz", placeholder="Enter student name here...")

col1, col2 = st.columns(2)

with col1:
    # Requirement 2: A study-hours input
    study_hours = st.number_input(
        "⏱️ Daily Study Hours:",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5,
        help="Number of hours spent studying each day"
    )

with col2:
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
predict_button = st.button("🚀 Predict Outcome Now", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

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

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-container">', unsafe_allow_html=True)

        # Requirement 4: A result message
        if prediction == 1:
            st.success(f"### 🎉 Congratulations, {student_name}!")
            st.success(f"**Predicted Class: PASS** 🟢 (Confidence: {probability:.1%})")
            st.balloons()
        else:
            st.error(f"### ⚠️ Warning, {student_name}!")
            st.error(f"**Predicted Class: FAIL** 🔴 (Pass Probability: {probability:.1%})")

        # Requirement 5: A short explanation of how the prediction was generated
        with st.expander("ℹ️ How was this prediction generated?", expanded=True):
            st.markdown(f"""
            **Explanation & Breakdown:**
            - **Input Features:** **{study_hours}** study hours/day and **{attendance}%** attendance.
            - **Algorithm:** Logistic Regression classifier loaded securely via `joblib`.
            - **Decision Logic:** The model computes the linear formula:
              $$\\text{{Score}} = (w_1 \\times \\text{{Study Hours}}) + (w_2 \\times \\text{{Attendance}}) + b$$
            - The score is processed using the **Sigmoid function** to calculate the exact percentage.
            - Result: With a probability of **{probability:.1%}** (against the 50% threshold), the model classifies this as **{"PASS" if prediction == 1 else "FAIL"}**.
            """)

        # Requirement 6 (BONUS): Add a chart that compares the study hours with the predicted class
        st.divider()
        st.subheader("📊 Comparison Chart: Study Hours vs. Predicted Class")
        
        # Historical reference data
        chart_data = pd.DataFrame({
            'Study Hours': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0],
            'Outcome': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 0: Fail, 1: Pass
        })

        # Styling Matplotlib for modern dark/light integration
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(8, 4))

        fail_data = chart_data[chart_data['Outcome'] == 0]
        pass_data = chart_data[chart_data['Outcome'] == 1]
        
        ax.scatter(fail_data['Study Hours'], fail_data['Outcome'], color='#ff4b4b', label='Historical: Fail (0)', s=70, alpha=0.7, edgecolor='k')
        ax.scatter(pass_data['Study Hours'], pass_data['Outcome'], color='#21c354', label='Historical: Pass (1)', s=70, alpha=0.7, edgecolor='k')

        # Highlight current student (Blue Glowing Star equivalent)
        ax.scatter([study_hours], [prediction], color='#1c83e1', s=250, marker='*', label=f'Current: {student_name}', zorder=5, edgecolor='black', linewidth=0.8)

        # Draw decision boundary threshold line at 4.0 hours
        ax.axvline(x=4.0, color='#888888', linestyle='--', linewidth=1.8, label='Pass Threshold (4.0 hrs)')

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Fail (0)', 'Pass (1)'], fontsize=10, fontweight='bold')
        ax.set_xlabel('Daily Study Hours', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Class', fontsize=11, fontweight='bold')
        ax.set_title('Study Hours vs. Student Outcome Analysis', fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

        st.pyplot(fig)
        st.caption("⭐ *The blue star highlights your input position compared against historical pass/fail distribution records.*")
        st.markdown('</div>', unsafe_allow_html=True)