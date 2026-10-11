
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load the existing trained model
model = joblib.load("student_pass_model.joblib")

# Deep blue UI theme
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #06152E 0%, #0B2452 55%, #103B79 100%);
    color: #F1F7FF;
}

/* Hide Streamlit top header background */
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

/* Main content width and spacing */
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main headings */
h1, h2, h3 {
    color: #FFFFFF !important;
}

p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: #D7E8FF;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #0756C9, #168BFF, #42B7FF);
    padding: 35px 32px;
    border-radius: 22px;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(0, 105, 255, 0.25);
    border: 1px solid rgba(255,255,255,0.2);
}

.hero h1 {
    font-size: 36px;
    margin: 0;
    font-weight: 800;
    color: white !important;
}

.hero p {
    color: #EAF5FF;
    font-size: 16px;
    margin: 10px 0 0 0;
}

/* Section cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(12, 42, 88, 0.88);
    border: 1px solid #2865B4;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
}

/* Input labels */
.stTextInput label,
.stNumberInput label {
    color: #C8E2FF !important;
    font-weight: 600;
}

/* Input fields */
.stTextInput input,
.stNumberInput input {
    background-color: #071B3A !important;
    color: #FFFFFF !important;
    border: 1px solid #347FE0 !important;
    border-radius: 10px !important;
}

/* Focused input */
.stTextInput input:focus,
.stNumberInput input:focus {
    border: 1px solid #55B7FF !important;
    box-shadow: 0 0 0 2px rgba(44, 145, 255, 0.2) !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #0878F9, #21B1FF);
    color: white !important;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 750;
    width: 100%;
    transition: 0.2s ease;
    box-shadow: 0 5px 18px rgba(0, 123, 255, 0.25);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #2595FF, #56CCFF);
    color: #FFFFFF !important;
    border: 1px solid #9ADDFF;
    transform: translateY(-2px);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0D3470, #1253A0);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #347FE0;
}

[data-testid="stMetricLabel"] {
    color: #B9DBFF !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #2865B4;
}

/* Footer */
.footer {
    text-align: center;
    color: #8EBBEF;
    font-size: 13px;
    padding-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Predictor</h1>
    <p>
        SMART ACADEMIC INSIGHTS · Study habits · Attendance · Predicted results
    </p>
</div>
""", unsafe_allow_html=True)

# Two-column main layout
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    with st.container(border=True):
        st.subheader("📘 Student Information")
        st.caption("Complete all fields to generate a prediction.")

        with st.form("prediction_form"):
            student_name = st.text_input(
                "Student Full Name *",
                placeholder="Enter student's full name"
            )

            study_hours = st.number_input(
                "Daily Study Hours",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5
            )

            attendance = st.number_input(
                "Attendance Percentage (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

            submitted = st.form_submit_button(
                "✨  PREDICT STUDENT RESULT",
                use_container_width=True
            )

with right:
    with st.container(border=True):
        st.subheader("📊 Input Overview")
        st.caption("Your entered academic information")

        st.metric("Daily Study Hours", f"{study_hours:g} hours")
        st.progress(min(study_hours / 24, 1.0))

        st.metric("Attendance Rate", f"{attendance:g}%")
        st.progress(min(attendance / 100, 1.0))

        st.markdown("---")
        st.markdown("**Features used by the model**")
        st.markdown("🔹 Study hours per day")
        st.markdown("🔹 Attendance percentage")

# Prediction
if submitted:
    if not student_name.strip():
        st.error("⚠️ Student name is required. Please enter the student's name.")
    else:
        student = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance_percentage": attendance
        }])

        prediction = model.predict(student)[0]

        st.markdown("---")

        with st.container(border=True):
            st.subheader("🎯 Prediction Results")
            st.markdown(
                f"**Student:** {student_name.strip()}"
            )

            if prediction == "Pass":
                st.success("### PASS — The model predicts a passing result.")
            else:
                st.error("### FAIL — The model predicts a failing result.")

            st.info(
                "The prediction is based on study hours and attendance. "
                "It is an estimate, not a guarantee of academic performance."
            )

        # Blue-themed chart
        with st.container(border=True):
            st.subheader("📈 Student Input Visualization")

            fig, ax = plt.subplots(figsize=(9, 3.5))
            fig.patch.set_facecolor("#0B2452")
            ax.set_facecolor("#0D3470")

            bars = ax.bar(
                ["Study Hours", "Attendance (%)"],
                [study_hours, attendance],
                color=["#168BFF", "#42C8FF"],
                width=0.5
            )

            ax.set_ylabel("Value", color="white")
            ax.set_title(
                "Academic Input Summary",
                color="white",
                fontsize=14,
                fontweight="bold"
            )
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="#D7E8FF")

            for spine in ax.spines.values():
                spine.set_color("#2865B4")

            ax.bar_label(bars, padding=4, color="white")
            ax.grid(axis="y", alpha=0.15, color="white")
            ax.set_axisbelow(True)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# Footer
st.markdown("""
<div class="footer">
    STUDENT PERFORMANCE PREDICTION SYSTEM · POWERED BY MACHINE LEARNING
</div>
""", unsafe_allow_html=True)
