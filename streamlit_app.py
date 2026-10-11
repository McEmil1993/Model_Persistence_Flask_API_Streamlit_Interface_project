import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Page Config
st.set_page_config(page_title="STUDENT PASS PREDICTOR", page_icon="🎓", layout="centered")

# 2. Custom Styling (Midnight Neon Theme)
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0B0F19; }
    
    /* Headers */
    h1, h2, h3 { color: #10B981 !important; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Buttons */
    div.stButton > button {
        background-color: #10B981 !important;
        color: #0B0F19 !important;
        border: 2px solid #3B82F6 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border-color: #10B981 !important;
        transform: scale(1.02);
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #374151;
        border-left: 4px solid #10B981;
        border-radius: 8px;
        padding: 15px;
    }
    div[data-testid="stMetricLabel"] { color: #9CA3AF !important; }
    div[data-testid="stMetricValue"] { color: #F3F4F6 !important; }
    </style>
""", unsafe_allow_html=True)

# 3. Load Model
MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Model file not found. Please run `train_model.py` first.")
    st.stop()

# 4. Header
st.title("Student Performance Prediction System")
st.write("Predict a student's result based on daily study hours using a trained Decision Tree model.")
st.divider()

# 5. Inputs
st.subheader("Student Information")
with st.form("prediction_form"):
    student_name = st.text_input("Student Name", placeholder="Enter name")
    study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    submitted = st.form_submit_button("Predict Result", use_container_width=True)

# 6. Prediction & Display
if submitted:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        sample = pd.DataFrame({"study_hours": [study_hours]})
        prediction = model.predict(sample)[0]
        
        # Calculate probabilities for the new graph
        try:
            proba = model.predict_proba(sample)[0]
            classes = model.classes_
        except AttributeError:
            proba = [0, 0]
            classes = ["Fail", "Pass"]
            st.warning("Probability data not available for this model.")

        st.divider()
        st.subheader("Prediction Result")
        st.write(f"**Student:** {student_name.strip()}")

        col1, col2 = st.columns(2)
        col1.metric("Predicted Result", prediction.upper())
        col2.metric("Study Hours", f"{study_hours:g} hours")

        if prediction == "Pass":
            st.success("The model predicts: PASS")
        else:
            st.warning("The model predicts: FAIL")

        # Explanation
        st.subheader("How was this generated?")
        st.info("The saved Decision Tree model uses patterns learned from the training dataset to predict a result based on study hours. The chart below shows the model's confidence in its prediction.")

        # 7. NEW BONUS CHART: Horizontal Bar Chart for Probabilities
        st.divider()
        st.subheader("Prediction Confidence")
        
        # Create the new graph
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#0B0F19")
        ax.set_facecolor("#111827")

        # Colors: Red for Fail, Green for Pass
        colors = ['#EF4444' if c == 'Fail' else '#10B981' for c in classes]
        
        # Plot Horizontal Bars
        bars = ax.barh(classes, proba, color=colors, height=0.5, edgecolor='#374151')
        
        # Add percentage text at the end of each bar
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{width*100:.1f}%', 
                    va='center', ha='left', color='#F3F4F6', fontweight='bold')

        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Probability", color="#9CA3AF")
        ax.set_title(f"Model Confidence for {student_name.strip()}", color="#10B981", pad=15)
        
        ax.tick_params(colors="#F3F4F6")
        ax.grid(color="#374151", axis='x', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color("#374151")

        st.pyplot(fig)