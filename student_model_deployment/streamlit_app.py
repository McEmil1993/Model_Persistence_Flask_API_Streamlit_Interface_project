import streamlit as st
import joblib
import pandas as pd

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Student Performance System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("student_result_model.joblib")

try:
    model = load_model()
    model_error = False
except Exception:
    model = None
    model_error = True

# ==========================================
# PINK AND LAVENDER DESIGN
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #fff0f7 0%, #f8efff 50%, #fff7fb 100%);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(115deg, #ff69b4, #e879f9, #c084fc);
    padding: 35px;
    border-radius: 25px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(217, 70, 159, 0.20);
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: white;
    font-size: 15px;
    margin-bottom: 0;
}

.section-card {
    background: rgba(255, 255, 255, 0.90);
    border: 1px solid #f5c9e8;
    border-radius: 22px;
    padding: 23px;
    margin-bottom: 20px;
    box-shadow: 0 5px 18px rgba(190, 100, 170, 0.09);
}

.section-title {
    color: #76276f;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 18px;
}

.metric-card {
    background: linear-gradient(135deg, #fff0f8, #f5eaff);
    border: 1px solid #f5d0ed;
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    min-height: 125px;
}

.metric-label {
    color: #81517f;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #e83e9c;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.result-box {
    background: linear-gradient(135deg, #ffe1f1, #f2ddff);
    border: 1px solid #f3b8df;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    margin-top: 15px;
}

.result-box h2 {
    color: #a21caf;
    font-size: 28px;
    font-weight: 800;
}

.result-box p {
    color: #8a347c;
}

.note {
    background: #f1efff;
    border-left: 5px solid #b477ec;
    border-radius: 10px;
    padding: 13px;
    color: #594078;
    font-size: 13px;
    margin-top: 14px;
}

.motivation {
    background: linear-gradient(135deg, #ffe0f0, #eadbff);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    color: #a04491;
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: #a14a94;
    padding: 22px;
    font-size: 13px;
}

div.stButton > button {
    background: linear-gradient(100deg, #ff4fa3, #d946ef);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 15px;
    font-size: 17px;
    font-weight: 700;
    min-height: 55px;
    width: 100%;
    transition: 0.2s;
}

div.stButton > button:hover {
    color: white;
    border: 1px solid #ec4899;
    box-shadow: 0 5px 18px rgba(236, 72, 153, 0.25);
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 12px;
    border-color: #e9b8dc;
}

label {
    color: #71376e !important;
    font-weight: 600 !important;
}

hr {
    border-color: #f3d4ed;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="hero">
    <h1>🎀 Student Performance Prediction System</h1>
    <p>🌸 Better Study Habits ✨ Brighter Future 🌸</p>
    <p>Welcome! Every little effort brings you closer to your dreams. 💗</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# ATTENDANCE ESTIMATION
# ==========================================
def calculate_attendance(study_hours):
    # Demo estimate only; this is not actual attendance data.
    return round(min(max(study_hours / 12 * 100, 0), 100))

# ==========================================
# MAIN DASHBOARD
# ==========================================
left_col, right_col = st.columns([0.85, 1.15], gap="large")

with left_col:

    st.markdown("""
    <div class="section-card">
        <div class="section-title">🌷 01 / Student Information</div>
    </div>
    """, unsafe_allow_html=True)

    student_name = st.text_input(
        "💗 Student Name",
        placeholder="Enter student name",
        key="student_name"
    )

    study_hours = st.number_input(
        "⏰ Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5,
        help="Enter the number of hours spent studying."
    )

    attendance = calculate_attendance(study_hours)

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">💜 Performance Overview</div>
        <div class="metric-card">
            <div class="metric-label">📚 Study Hours</div>
            <div class="metric-value">{study_hours:.1f} hrs</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-label">👭 Estimated Attendance</div>
            <div class="metric-value" style="color:#a855f7">{attendance}%</div>
        </div>
        <br>
        <div style="background:#f0e8ff;border-radius:20px;height:12px;overflow:hidden;">
            <div style="background:linear-gradient(90deg,#ff4fa3,#b56cf6);
                        width:{attendance}%;height:12px;border-radius:20px;">
            </div>
        </div>
        <p style="color:#865b91;font-size:12px;margin-top:10px;">
            Attendance estimate updates automatically when study hours change.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="motivation">
        💕 You can do it!<br>
        <span style="font-size:14px;font-weight:400;">
        Your effort today builds your success tomorrow.
        </span>
    </div>
    """, unsafe_allow_html=True)

    predict_clicked = st.button(
        "✨ PREDICT STUDENT RESULT ✨",
        use_container_width=True
    )

with right_col:

    st.markdown("""
    <div class="section-card">
        <div class="section-title">💌 02 / Prediction Report</div>
    </div>
    """, unsafe_allow_html=True)

    if predict_clicked:

        if not student_name.strip():
            st.warning("🌸 Please enter the student's name first.")

        elif model_error:
            st.error(
                "Please make sure "
                "'student_result_model.joblib' is in the project folder."
            )

        else:
            try:
                # Preserve the existing model: prediction uses study hours.
                prediction = model.predict([[study_hours]])[0]
                prediction_text = str(prediction).strip()

                st.session_state["prediction_report"] = {
                    "name": student_name.strip(),
                    "hours": study_hours,
                    "attendance": attendance,
                    "prediction": prediction_text
                }

            except Exception as e:
                st.error(f"Prediction error: {e}")

    report = st.session_state.get("prediction_report")

    if report:
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">🌸 Student Details</div>
            <p><b>👤 Student Name:</b> {report['name']}</p>
            <p><b>📚 Study Hours:</b> {report['hours']:.1f} hours</p>
            <p><b>💜 Estimated Attendance:</b> {report['attendance']}%</p>
            <hr>
            <div class="result-box">
                <p>🎯 PREDICTED STUDENT RESULT</p>
                <h2>{report['prediction']}</h2>
                <p>Keep learning and doing your best! 🌷</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.success("💖 Prediction completed successfully!")

    else:
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">🌸 Your Prediction Report</div>
            <p><b>👤 Student Name:</b> {student_name.strip() or '—'}</p>
            <p><b>📚 Study Hours:</b> {study_hours:.1f} hours</p>
            <p><b>💜 Estimated Attendance:</b> {attendance}%</p>
            <div class="result-box">
                <p>🎯 PREDICTED STUDENT RESULT</p>
                <h2>—</h2>
                <p>Click the pink button to see your result! 💕</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">📊 03 / Student Overview Chart</div>
    </div>
    """, unsafe_allow_html=True)

    chart_data = pd.DataFrame({
        "Performance Metric": ["Study Hours", "Estimated Attendance (%)"],
        "Value": [study_hours, attendance]
    }).set_index("Performance Metric")

    st.bar_chart(chart_data)

    st.markdown("""
    <div class="note">
        💡 <b>Note:</b> Attendance is an estimated demo value calculated
        from study hours, not the student's actual attendance record.
        The trained model predicts the result using study hours only.
        Attendance does not affect the model's prediction.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    💗 Student Performance Prediction System · Girl Version 🎀<br>
    Made with love, dedication, and a little sparkle ✨
</div>
""", unsafe_allow_html=True)