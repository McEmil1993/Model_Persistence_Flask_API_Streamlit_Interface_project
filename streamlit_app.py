
import streamlit as st
import joblib
import pandas as pd

# Load model once — unchanged
model = joblib.load("student_pass_model.joblib")

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Glassmorphism UI Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at top left, rgba(99, 102, 241, 0.35), transparent 45%),
        radial-gradient(ellipse at bottom right, rgba(14, 165, 233, 0.25), transparent 45%),
        linear-gradient(135deg, #0f172a, #111827 55%, #172554);
    font-family: 'Inter', sans-serif;
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1100px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Main glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.075);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 24px;
    padding: 30px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.22);
    margin-bottom: 24px;
}

.hero {
    text-align: center;
    padding: 35px 20px;
    background: linear-gradient(
        135deg,
        rgba(99, 102, 241, 0.20),
        rgba(14, 165, 233, 0.10)
    );
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 28px;
    margin-bottom: 28px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

.hero-icon {
    font-size: 48px;
    margin-bottom: 8px;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(27px, 4vw, 42px);
    font-weight: 800;
    margin: 0;
    letter-spacing: -1.2px;
}

.hero p {
    color: #cbd5e1;
    font-size: 15px;
    margin-top: 12px;
}

.section-title {
    color: #ffffff;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 22px;
}

/* Input labels */
.stApp label,
.stMarkdown p {
    color: #e2e8f0;
}

.stTextInput input,
.stNumberInput input {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.20) !important;
    border-radius: 12px !important;
    min-height: 45px;
}

.stTextInput input::placeholder {
    color: #94a3b8 !important;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.18) !important;
}

.stSlider [data-baseweb="slider"] {
    margin-top: 12px;
}

.stSlider [data-testid="stTickBar"] {
    background: rgba(255, 255, 255, 0.2);
}

/* Predict button */
.stButton > button {
    width: 100%;
    min-height: 50px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 14px;
    background: linear-gradient(135deg, #6366f1, #0ea5e9);
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    transition: all 0.25s ease;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.25);
}

.stButton > button:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.45);
    box-shadow: 0 10px 28px rgba(79, 70, 229, 0.4);
    color: #ffffff;
}

/* Result panel */
.result-pass {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(74, 222, 128, 0.35);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin-top: 20px;
}

.result-fail {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(248, 113, 113, 0.35);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin-top: 20px;
}

.result-label {
    font-size: 12px;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}

.result-value {
    font-size: 42px;
    font-weight: 800;
    margin-top: 8px;
}

.result-name {
    color: #e2e8f0;
    font-size: 16px;
    margin-top: 8px;
}

/* Info messages */
.stAlert {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 14px;
}

/* Chart container */
div[data-testid="stArrowVegaLiteChart"],
div[data-testid="stVegaLiteChart"],
div[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 16px;
    padding: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding-top: 25px;
}

#MainMenu, footer {
    visibility: hidden;
}

@media (max-width: 640px) {
    .block-container {
        padding: 1.2rem 1rem;
    }

    .glass-card {
        padding: 20px;
    }

    .hero {
        padding: 28px 14px;
    }
}
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown("""
<div class="hero">
    <div class="hero-icon">🎓</div>
    <h1>Student Pass Predictor</h1>
    <p>Smart academic insights powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# --- Two-column layout ---
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">📝 Student Information</div>
        <div class="section-subtitle">
            Enter the student's details to generate a prediction.
        </div>
    """, unsafe_allow_html=True)

    # Inputs — unchanged
    student_name = st.text_input(
        "Student Name",
        placeholder="e.g., Juan Dela Cruz"
    )

    study_hours = st.number_input(
        "Study Hours per Week",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )

    attendance = st.slider(
        "Attendance Percentage",
        min_value=0,
        max_value=100,
        value=75
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    predict_clicked = st.button("✨  Predict Student Result")

with right_col:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">📊 Prediction Result</div>
        <div class="section-subtitle">
            Your prediction and study information will appear here.
        </div>
    """, unsafe_allow_html=True)

    if predict_clicked:
        if not student_name.strip():
            st.warning("Please enter the student's name.")
        else:
            # Prediction logic — unchanged
            pred = model.predict([[study_hours, attendance]])[0]
            label = "Pass" if pred == 1 else "Fail"
            color = "green" if pred == 1 else "red"

            result_class = "result-pass" if pred == 1 else "result-fail"

            st.markdown(f"""
            <div class="{result_class}">
                <div class="result-label">Prediction for</div>
                <div class="result-name">{student_name}</div>
                <div class="result-value" style="color:{color};">
                    {"✓" if pred == 1 else "✕"} {label}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Explanation — unchanged
            st.info(
                f"The model used a Logistic Regression classifier trained on "
                f"study hours and attendance data. With **{study_hours} study hours** "
                f"and **{attendance}% attendance**, it predicted **{label}**."
            )

            # Chart — same data and prediction
            st.markdown("#### 📈 Study Hours vs Predicted Class")

            chart_data = pd.DataFrame({
                "Study Hours": [study_hours],
                "Predicted Class (1=Pass, 0=Fail)": [int(pred)]
            })

            st.bar_chart(chart_data.set_index("Study Hours"))

    else:
        st.markdown("""
        <div style="
            padding: 40px 15px;
            text-align: center;
            border: 1px dashed rgba(255,255,255,0.2);
            border-radius: 18px;
            background: rgba(255,255,255,0.025);
        ">
            <div style="font-size: 38px;">🔮</div>
            <p style="color: #cbd5e1; font-weight: 600;">
                Waiting for prediction
            </p>
            <p style="color: #94a3b8; font-size: 13px;">
                Fill in the student information and click Predict.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --- Footer ---
st.markdown("""
<div class="footer">
    STUDENT PASS PREDICTOR · MACHINE LEARNING PROJECT
</div>
""", unsafe_allow_html=True)
