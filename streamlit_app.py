
import streamlit as st
import requests
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Student Analytics | Pass Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:5000/predict"

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b1120;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #111a2e;
        border-right: 1px solid #26334c;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .brand {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #60a5fa;
        margin-bottom: 12px;
    }

    .hero {
        padding: 28px 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172747, #14213a 65%, #123047);
        border: 1px solid #2c4265;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: clamp(27px, 3vw, 38px);
        font-weight: 800;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #b8c7dc;
        font-size: 15px;
        margin: 0;
        line-height: 1.8;
    }

    .section-heading {
        font-size: 19px;
        font-weight: 700;
        color: #f1f5f9;
        margin: 12px 0 5px 0;
    }

    .section-description {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .panel {
        background: #111a2e;
        border: 1px solid #263650;
        padding: 23px;
        border-radius: 15px;
        margin-bottom: 18px;
    }

    .metric-card {
        background: #111a2e;
        border: 1px solid #263650;
        border-radius: 14px;
        padding: 19px;
        min-height: 112px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 27px;
        font-weight: 800;
        margin-top: 9px;
    }

    .metric-note {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 4px;
    }

    .result-pass {
        padding: 25px;
        border-radius: 15px;
        background: #102e29;
        border: 1px solid #20745c;
        margin: 12px 0 20px 0;
    }

    .result-fail {
        padding: 25px;
        border-radius: 15px;
        background: #351c29;
        border: 1px solid #87405b;
        margin: 12px 0 20px 0;
    }

    .result-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #cbd5e1;
        text-transform: uppercase;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        margin: 8px 0;
        color: #f8fafc;
    }

    .result-detail {
        color: #d1dbe8;
        font-size: 14px;
        line-height: 1.7;
    }

    .footer {
        border-top: 1px solid #263650;
        padding-top: 18px;
        margin-top: 35px;
        color: #64748b;
        font-size: 12px;
        text-align: center;
    }

    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 43px;
    }

    div[data-testid="stForm"] {
        background: #111a2e;
        border: 1px solid #263650;
        border-radius: 15px;
        padding: 22px;
    }

    label, [data-testid="stWidgetLabel"] p {
        color: #dbe5f3 !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #19253b;
        border-color: #354661;
        border-radius: 8px;
    }

    .stNumberInput button {
        color: #f1f5f9 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 22px 18px;
        }
        .metric-value {
            font-size: 22px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "api_error" not in st.session_state:
    st.session_state.api_error = None

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand">🎓 RARO STUDENT ANALYTICS</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Pass Predictor")
    st.caption("Academic performance support system")

    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠  Prediction Dashboard")
    st.markdown("📊  Student Assessment")
    st.markdown("📄  Prediction Report")

    st.divider()

    st.caption("Student Analytics System · Local deployment")

result = st.session_state.prediction_result

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">System status</div>
        <div class="metric-value" style="font-size:22px;">Ready</div>
        <div class="metric-note">Prediction interface active</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    student_count = 1 if result else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest assessments</div>
        <div class="metric-value">{student_count}</div>
        <div class="metric-note">Results in this session</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    latest_status = result["prediction"] if result else "No result"
    status_color = (
        "#34d399" if latest_status == "Pass"
        else "#fb7185" if latest_status == "Fail"
        else "#f8fafc"
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest prediction</div>
        <div class="metric-value" style="font-size:22px;color:{status_color};">
            {latest_status}
        </div>
        <div class="metric-note">Most recent assessment</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.markdown('<div class="section-heading">Student Assessment</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">'
    'Provide the student details and academic indicators below.'
    '</div>',
    unsafe_allow_html=True
)

with st.form("student_prediction_form"):
    student_name = st.text_input(
        "Student full name",
        placeholder="e.g. Krishia Mae Raro",
        help="Enter the name to display in the prediction report."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        study_hours = st.number_input(
            "Study hours per day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
            help="Average number of hours spent studying each day."
        )

    with col_b:
        attendance = st.number_input(
            "Attendance percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=1.0,
            help="The student's attendance percentage."
        )

    submitted = st.form_submit_button(
        "Generate Prediction →",
        type="primary",
        use_container_width=True
    )

    if submitted:
        if not student_name.strip():
            st.session_state.api_error = (
                "Please enter the student's name before proceeding."
            )
            st.session_state.prediction_result = None

        else:
            st.session_state.api_error = None

            try:
                with st.spinner("Analyzing student information..."):
                    response = requests.post(
                        API_URL,
                        json={
                            "study_hours": study_hours,
                            "attendance_percentage": attendance
                        },
                        timeout=10
                    )

                if response.ok:
                    data = response.json()
                    prediction = str(data.get("prediction", "")).strip().title()

                    if prediction not in ("Pass", "Fail"):
                        st.session_state.api_error = (
                            "The API returned an unexpected prediction. "
                            "Expected 'Pass' or 'Fail'. Check app.py."
                        )
                        st.session_state.prediction_result = None
                    else:
                        st.session_state.prediction_result = {
                            "name": student_name.strip(),
                            "study_hours": study_hours,
                            "attendance": attendance,
                            "prediction": prediction,
                            "timestamp": datetime.now().strftime(
                                "%B %d, %Y at %I:%M %p"
                            )
                        }

                else:
                    st.session_state.api_error = (
                        f"API error ({response.status_code}): "
                        f"{response.text[:500]}"
                    )
                    st.session_state.prediction_result = None

            except requests.exceptions.ConnectionError:
                st.session_state.api_error = (
                    "Unable to connect to the Flask API. "
                    "Make sure your backend is running on port 5000."
                )
                st.session_state.prediction_result = None

            except requests.exceptions.Timeout:
                st.session_state.api_error = (
                    "The prediction request timed out. "
                    "Check whether the Flask API is responding."
                )
                st.session_state.prediction_result = None

            except (requests.exceptions.RequestException, ValueError) as exc:
                st.session_state.api_error = (
                    f"Unable to process the prediction: {exc}"
                )
                st.session_state.prediction_result = None

# --------------------------------------------------
# ERRORS
# --------------------------------------------------
if st.session_state.api_error:
    st.error(st.session_state.api_error)

# --------------------------------------------------
# PREDICTION RESULTS
# --------------------------------------------------
result = st.session_state.prediction_result

if result:
    st.markdown("---")
    st.markdown(
        '<div class="section-heading">Prediction Report</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-description">'
        'Review the generated result and the information used for assessment.'
        '</div>',
        unsafe_allow_html=True
    )

    if result["prediction"] == "Pass":
        st.markdown(f"""
        <div class="result-pass">
            <div class="result-label">Assessment outcome</div>
            <div class="result-title">🎉 Predicted to Pass</div>
            <div class="result-detail">
                Student: <strong>{result["name"]}</strong><br>
                The current machine learning model predicts a passing result.
                This is an estimate, not a guarantee of academic performance.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-fail">
            <div class="result-label">Assessment outcome</div>
            <div class="result-title">📘 Predicted to Fail</div>
            <div class="result-detail">
                Student Name: <strong>{result["name"]}</strong><br>
                Consider reviewing the student's study habits and providing
                appropriate academic support. This prediction is not a
                definitive assessment of the student's abilities.
            </div>
        </div>
        """, unsafe_allow_html=True)

    metric1, metric2 = st.columns(2)

    with metric1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Study hours per day</div>
            <div class="metric-value">{result["study_hours"]:.1f} hrs</div>
            <div class="metric-note">Reported daily study time</div>
        </div>
        """, unsafe_allow_html=True)

    with metric2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Attendance</div>
            <div class="metric-value">{result["attendance"]:.0f}%</div>
            <div class="metric-note">Recorded attendance percentage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Student indicators")

    st.write("**Study hours per day**")
    st.progress(min(result["study_hours"] / 24.0, 1.0))
    st.caption(f'{result["study_hours"]:.1f} out of 24 hours')

    st.write("**Attendance percentage**")
    st.progress(result["attendance"] / 100.0)
    st.caption(f'{result["attendance"]:.0f}% attendance recorded')

    report = f"""STUDENT PASS PREDICTION REPORT
================================

Student Name: {result["name"]}
Prediction: {result["prediction"]}
Study Hours Per Day: {result["study_hours"]:.1f}
Attendance: {result["attendance"]:.1f}%
Generated: {result["timestamp"]}

MODEL INFORMATION
-----------------
Prediction service: Local Flask API
Endpoint: {API_URL}
Primary model input: Study hours per day
Attendance: Recorded for reference only

DISCLAIMER
----------
This result is a machine learning estimate, not a guarantee
of academic performance. The model currently uses study hours
only, and attendance does not affect the prediction.
"""

    st.download_button(
        label="📄 Download Student Report",
        data=report,
        file_name="student_prediction_report.txt",
        mime="text/plain",
        use_container_width=True
    )

else:
    st.markdown("""
    <div class="panel">
        <div class="section-heading">Your results will appear here</div>
        <div class="section-description" style="margin-bottom:0;">
            Complete the student assessment form and select
            <strong>Generate Prediction</strong> to view the report.
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    Student Analytics &nbsp;•&nbsp; Machine Learning Prediction System
    <br><br>
    For educational purposes. Predictions should be interpreted
    alongside other relevant academic information.
</div>
""", unsafe_allow_html=True)
