
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

# --- Dark Analytics UI ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0b1120;
    color: #e5e7eb;
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

small {
    color: #94a3b8;
}

/* Header */
.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
    padding: 26px 30px;
    background: linear-gradient(120deg, #111c30, #101827);
    border: 1px solid #263449;
    border-radius: 18px;
    margin-bottom: 28px;
}

.dashboard-title {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1px;
}

.dashboard-subtitle {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 7px;
}

.status-pill {
    display: inline-block;
    padding: 8px 13px;
    background: rgba(34, 197, 94, 0.10);
    border: 1px solid rgba(34, 197, 94, 0.25);
    border-radius: 30px;
    color: #4ade80;
    font-size: 12px;
    font-weight: 600;
}

/* Dashboard cards */
.panel {
    background: #111c30;
    border: 1px solid #263449;
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 22px;
}

.panel-title {
    color: #f1f5f9;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 6px;
}

.panel-description {
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.6;
    margin-bottom: 22px;
}

.metric-card {
    background: #111c30;
    border: 1px solid #263449;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 12px;
}

.metric-value {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 800;
}

.metric-accent {
    color: #4ade80;
}

/* Inputs */
.stTextInput input,
.stNumberInput input {
    background: #0b1425 !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 9px !important;
    min-height: 44px;
}

.stTextInput input::placeholder {
    color: #64748b !important;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 1px #22c55e !important;
}

.stSlider [data-baseweb="slider"] {
    padding-top: 8px;
    padding-bottom: 8px;
}

.stSlider [data-testid="stTickBar"] {
    background: #334155;
}

/* Slider value */
.stSlider [data-testid="stThumbValue"] {
    color: #4ade80;
}

/* Buttons */
.stButton > button {
    width: 100%;
    min-height: 46px;
    background: #16a34a;
    color: #ffffff;
    border: 1px solid #22c55e;
    border-radius: 9px;
    font-weight: 700;
    font-size: 14px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #15803d;
    border-color: #4ade80;
    color: #ffffff;
    box-shadow: 0 0 18px rgba(34, 197, 94, 0.15);
}

/* Prediction result */
.result-card {
    border-radius: 12px;
    padding: 25px;
    margin: 18px 0;
    text-align: center;
}

.result-pass {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.35);
}

.result-fail {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.result-caption {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.result-name {
    color: #e2e8f0;
    font-size: 16px;
    margin-top: 10px;
}

.result-value {
    font-size: 38px;
    font-weight: 800;
    margin-top: 10px;
}

/* Alerts */
.stAlert {
    background: #172338;
    border: 1px solid #334155;
    border-radius: 10px;
    color: #e2e8f0;
}

/* Chart area */
div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {
    background: #0b1425;
    border: 1px solid #263449;
    border-radius: 10px;
    padding: 10px;
}

/* Divider */
hr {
    border-color: #263449;
}

/* Footer */
.dashboard-footer {
    text-align: center;
    color: #64748b;
    font-size: 11px;
    letter-spacing: 1px;
    padding-top: 18px;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1.2rem 1rem;
    }

    .dashboard-header {
        padding: 20px;
    }

    .dashboard-title {
        font-size: 24px;
    }

    .panel {
        padding: 18px;
    }

    .metric-card {
        padding: 15px;
    }
}
</style>
""", unsafe_allow_html=True)


# --- Dashboard Header ---
st.markdown("""
<div class="dashboard-header">
    <div>
        <div class="dashboard-title">🎓 Student Analytics</div>
        <div class="dashboard-subtitle">
            Student Pass Prediction · Machine Learning Dashboard
        </div>
    </div>
    <div class="status-pill">● MODEL READY</div>
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
        <div class="metric-value">02</div>
    </div>
    """, unsafe_allow_html=True)

with metric3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">PREDICTION CLASSES</div>
        <div class="metric-value metric-accent">Pass / Fail</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# --- Main Dashboard Layout ---
left_col, right_col = st.columns([1, 1.15], gap="large")

with left_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">📝 Student Information</div>
        <div class="panel-description">
            Enter student details to run the machine learning prediction.
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

    predict_clicked = st.button("▶  Run Prediction", use_container_width=True)


with right_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">📊 Prediction Analysis</div>
        <div class="panel-description">
            View the predicted academic outcome and supporting information.
        </div>
    """, unsafe_allow_html=True)

    if predict_clicked:
        if not student_name.strip():
            st.warning("Please enter the student's name.")
        else:
            # Prediction logic — unchanged
            pred = model.predict([[study_hours, attendance]])[0]
            label = "Pass" if pred == 1 else "Fail"
            color = "#4ade80" if pred == 1 else "#f87171"
            result_class = "result-pass" if pred == 1 else "result-fail"
            symbol = "✓" if pred == 1 else "✕"

            st.markdown(f"""
            <div class="result-card {result_class}">
                <div class="result-caption">Predicted Result</div>
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
            st.markdown("#### Study Hours vs Predicted Class")

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
            border: 1px dashed #334155;
            border-radius: 12px;
            background: #0b1425;
        ">
            <div style="font-size: 36px; margin-bottom: 12px;">📈</div>
            <div style="
                color: #e2e8f0;
                font-size: 15px;
                font-weight: 600;
            ">
                Awaiting Prediction
            </div>
            <div style="
                color: #94a3b8;
                font-size: 12px;
                margin-top: 8px;
            ">
                Enter student information and run a prediction
                to view the analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --- Footer ---
st.markdown("""
<div class="dashboard-footer">
    STUDENT ANALYTICS · MACHINE LEARNING PROJECT
</div>
""", unsafe_allow_html=True)