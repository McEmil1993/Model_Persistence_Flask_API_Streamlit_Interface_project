import streamlit as st
import joblib
import pandas as pd

# Load model once
model = joblib.load("student_pass_model.joblib")

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Gradient Modern UI Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(124, 58, 237, 0.15), transparent 35%),
        radial-gradient(circle at 100% 100%, rgba(6, 182, 212, 0.13), transparent 35%),
        #0f172a;
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1200px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Typography */
h1, h2, h3, h4 {
    color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
}

p, label {
    color: #cbd5e1;
}

/* Hero header */
.gradient-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, #7c3aed 0%, #4f46e5 48%, #0891b2 100%);
    border-radius: 24px;
    padding: 35px;
    margin-bottom: 28px;
    box-shadow: 0 16px 40px rgba(79, 70, 229, 0.22);
}

.hero-eyebrow {
    color: #e0e7ff;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.5px;
    margin-bottom: 12px;
}

.hero-title {
    color: #ffffff;
    font-size: clamp(28px, 4vw, 43px);
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.2;
}

.hero-subtitle {
    color: #e0e7ff;
    font-size: 14px;
    line-height: 1.8;
    margin-top: 12px;
    max-width: 650px;
}

.hero-badge {
    display: inline-block;
    margin-top: 20px;
    padding: 9px 14px;
    color: #ffffff;
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1e293b, #172033);
    border: 1px solid #334155;
    border-radius: 17px;
    padding: 22px;
    min-height: 120px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.12);
}

.metric-label {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.3px;
    margin-bottom: 13px;
}

.metric-value {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 800;
}

.metric-accent {
    background: linear-gradient(90deg, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Content panels */
.gradient-panel {
    background: linear-gradient(145deg, #1e293b, #151e30);
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 26px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
}

.panel-title {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 8px;
}

.panel-description {
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.7;
    margin-bottom: 23px;
}

/* Inputs */
.stTextInput input,
.stNumberInput input {
    background: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
    border-radius: 11px !important;
    min-height: 46px;
}

.stTextInput input::placeholder {
    color: #64748b !important;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.18) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    padding-top: 8px;
    padding-bottom: 8px;
}

/* Gradient button */
.stButton > button {
    width: 100%;
    min-height: 49px;
    color: #ffffff;
    background: linear-gradient(100deg, #7c3aed, #4f46e5, #0891b2);
    border: 1px solid rgba(167, 139, 250, 0.5);
    border-radius: 11px;
    font-size: 14px;
    font-weight: 800;
    transition: all 0.25s ease;
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.20);
}

.stButton > button:hover {
    color: #ffffff;
    border-color: #a5f3fc;
    background: linear-gradient(100deg, #8b5cf6, #6366f1, #06b6d4);
    box-shadow: 0 10px 28px rgba(79, 70, 229, 0.35);
    transform: translateY(-2px);
}

/* Prediction result */
.result-card {
    border-radius: 16px;
    padding: 28px 20px;
    margin: 18px 0;
    text-align: center;
}

.result-pass {
    background: linear-gradient(
        135deg,
        rgba(16, 185, 129, 0.13),
        rgba(6, 182, 212, 0.07)
    );
    border: 1px solid rgba(52, 211, 153, 0.40);
}

.result-fail {
    background: linear-gradient(
        135deg,
        rgba(244, 63, 94, 0.13),
        rgba(249, 115, 22, 0.07)
    );
    border: 1px solid rgba(251, 113, 133, 0.40);
}

.result-caption {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.result-name {
    color: #e2e8f0;
    font-size: 16px;
    margin-top: 12px;
}

.result-value {
    font-size: 40px;
    font-weight: 800;
    margin-top: 12px;
}

/* Alerts */
.stAlert {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
}

/* Chart */
div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 10px;
}

/* Footer */
.gradient-footer {
    text-align: center;
    color: #64748b;
    font-size: 11px;
    letter-spacing: 1.5px;
    padding-top: 22px;
}

hr {
    border-color: #334155;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .gradient-hero {
        padding: 25px 20px;
    }

    .gradient-panel {
        padding: 20px;
    }

    .metric-card {
        padding: 15px;
    }

    .metric-value {
        font-size: 17px;
    }

    .result-value {
        font-size: 32px;
    }
}
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown("""
<div class="gradient-hero">
    <div class="hero-eyebrow">SMART ACADEMIC ANALYTICS</div>
    <div class="hero-title">🎓 Student Pass Predictor</div>
    <div class="hero-subtitle">
        Transform student study habits and attendance into
        machine-learning-powered academic predictions.
    </div>
    <div class="hero-badge">✦ MACHINE LEARNING DASHBOARD</div>
</div>
""", unsafe_allow_html=True)


# --- Overview Metrics ---
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MODEL TYPE</div>
        <div class="metric-value">Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with metric2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">INPUT FEATURES</div>
        <div class="metric-value metric-accent">02</div>
    </div>
    """, unsafe_allow_html=True)

with metric3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">OUTPUT CLASSES</div>
        <div class="metric-value">Pass / Fail</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# --- Main Layout ---
left_col, right_col = st.columns([1, 1.15], gap="large")

with left_col:
    st.markdown("""
    <div class="gradient-panel">
        <div class="panel-title">📝 Student Information</div>
        <div class="panel-description">
            Provide the student's information to generate a prediction.
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

    predict_clicked = st.button(
        "✨  Predict Student Result",
        use_container_width=True
    )


with right_col:
    st.markdown("""
    <div class="gradient-panel">
        <div class="panel-title">📊 Prediction Results</div>
        <div class="panel-description">
            Review the model's prediction and supporting information.
        </div>
    """, unsafe_allow_html=True)

    if predict_clicked:
        if not student_name.strip():
            st.warning("Please enter the student's name.")
        else:
            # Prediction logic — unchanged
            pred = model.predict([[study_hours, attendance]])[0]
            label = "Pass" if pred == 1 else "Fail"
            color = "#34d399" if pred == 1 else "#fb7185"
            result_class = "result-pass" if pred == 1 else "result-fail"
            symbol = "✓" if pred == 1 else "✕"

            st.markdown(f"""
            <div class="result-card {result_class}">
                <div class="result-caption">PREDICTED OUTCOME</div>
                <div class="result-name">{student_name}</div>
                <div class="result-value" style="color:{color};">
                    {symbol} {label}
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
            text-align: center;
            padding: 42px 15px;
            background: linear-gradient(145deg, #172033, #1e293b);
            border: 1px solid #334155;
            border-radius: 15px;
        ">
            <div style="font-size: 40px; margin-bottom: 14px;">🚀</div>
            <div style="
                color: #f8fafc;
                font-size: 16px;
                font-weight: 700;
            ">
                Ready to Predict
            </div>
            <div style="
                color: #94a3b8;
                font-size: 12px;
                line-height: 1.8;
                margin-top: 10px;
            ">
                Enter the student's details and run the prediction
                to see the result here.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --- Footer ---
st.markdown("""
<div class="gradient-footer">
    STUDENT PASS PREDICTOR · MACHINE LEARNING PROJECT
</div>
""", unsafe_allow_html=True)