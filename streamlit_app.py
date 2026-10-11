"""
streamlit_app.py — Complete with Fixed Chart
Frontend that calls the Flask backend API.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS — DARK MODE
# ============================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: #0f172a !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    html, body, [class*="css"], .stMarkdown, p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }

    /* Hero Header */
    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        color: white !important;
        text-align: center;
        box-shadow: 0 20px 50px rgba(139, 92, 246, 0.5);
        margin-bottom: 2rem;
    }
    .hero-emoji {
        font-size: 4rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0;
        color: white !important;
        letter-spacing: -1px;
    }
    .hero p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-top: 0.75rem;
        margin-bottom: 0;
        color: white !important;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255,255,255,0.25);
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        margin-top: 1.2rem;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.4);
        font-weight: 600;
    }

    /* Section Title */
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #a5b4fc !important;
        margin: 0 0 1.2rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Inputs */
    .stTextInput label, .stNumberInput label {
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #334155 !important;
        padding: 0.7rem 1rem !important;
        font-size: 1rem !important;
        background: #1e293b !important;
        color: #f1f5f9 !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.2) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        border: none;
        padding: 0.85rem 2rem;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.05rem;
        width: 100%;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.5);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(139, 92, 246, 0.6);
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white !important;
    }

    /* Result Banner */
    .result-pass {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white !important;
        text-align: center;
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.5);
        margin-bottom: 1.5rem;
    }
    .result-fail {
        background: linear-gradient(135deg, #f43f5e 0%, #fb7185 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white !important;
        text-align: center;
        box-shadow: 0 15px 40px rgba(244, 63, 94, 0.5);
        margin-bottom: 1.5rem;
    }
    .result-emoji {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 0.4rem;
    }
    .result-pass h2, .result-fail h2 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 900;
        color: white !important;
    }
    .result-pass p, .result-fail p {
        margin: 0.6rem 0 0 0;
        font-size: 1rem;
        opacity: 0.95;
        color: white !important;
    }

    /* Metric Cards */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        padding: 1.3rem 1rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 2px solid #334155;
    }
    .metric-card:hover {
        border-color: #8b5cf6;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 800;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 900;
        color: #a5b4fc !important;
        margin-top: 0.4rem;
    }

    /* Info Box */
    .info-box {
        background: #1e293b;
        border-left: 5px solid #8b5cf6;
        padding: 1.3rem 1.5rem;
        border-radius: 14px;
        margin-top: 1rem;
        color: #e2e8f0 !important;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .info-box strong {
        color: #a5b4fc !important;
    }
    .info-box code {
        background: #0f172a;
        padding: 0.15rem 0.4rem;
        border-radius: 5px;
        color: #f472b6 !important;
        font-size: 0.85rem;
        border: 1px solid #334155;
    }

    /* Placeholder */
    .placeholder {
        background: #1e293b;
        padding: 3.5rem 2rem;
        border-radius: 20px;
        border: 2px dashed #475569;
        text-align: center;
        color: #94a3b8 !important;
    }
    .placeholder-emoji {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 0.8rem;
    }
    .placeholder p {
        font-size: 1.05rem;
        margin: 0;
        color: #cbd5e1 !important;
    }
    .placeholder strong {
        color: #a5b4fc !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b !important;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #334155;
        line-height: 1.8;
    }
    .footer strong {
        color: #a5b4fc !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <span class="hero-emoji">🎓</span>
    <h1>Student Pass Predictor</h1>
    <p>AI-powered prediction using Machine Learning</p>
    <div class="badge">⚡ Flask API + Logistic Regression</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUTS
# ============================================================
st.markdown('<div class="section-title">📝 Student Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    student_name = st.text_input(
        "👤 Student Name",
        placeholder="e.g. Juan Dela Cruz",
        key="name_input"
    )
    study_hours = st.number_input(
        "📚 Study Hours per Day",
        min_value=0.0, max_value=24.0, value=5.0, step=0.5
    )
with col2:
    attendance = st.number_input(
        "🗓️ Attendance Percentage",
        min_value=0.0, max_value=100.0, value=85.0, step=1.0
    )
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮 Predict Result", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# RESULTS
# ============================================================
if predict_clicked:
    if not student_name.strip():
        st.warning("⚠️ Please enter the student's name.")
    else:
        payload = {
            "student_name": student_name,
            "study_hours": study_hours,
            "attendance_percentage": attendance
        }
        try:
            with st.spinner("🔄 Contacting Flask API..."):
                response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()

                # Result Banner
                if result["prediction"] == "Pass":
                    st.markdown(f"""
                    <div class="result-pass">
                        <span class="result-emoji">🎉</span>
                        <h2>{result['student_name']} — PASS</h2>
                        <p>This student is predicted to pass the course.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-fail">
                        <span class="result-emoji">📘</span>
                        <h2>{result['student_name']} — FAIL</h2>
                        <p>This student is predicted to fail the course.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Metric Cards
                st.markdown(f"""
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Prediction</div>
                        <div class="metric-value">{result['prediction']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{result['confidence']}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Study Hours</div>
                        <div class="metric-value">{result['study_hours']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Explanation
                st.markdown(f"""
                <div class="info-box">
                    <strong>🧠 How this was generated:</strong><br>
                    Streamlit sent a POST request to the Flask API.
                    The backend loaded <code>student_pass_model.joblib</code> and
                    made a prediction using <strong>{result['study_hours']} study hours</strong>
                    and <strong>{result['attendance_percentage']}% attendance</strong>.
                    The model returned <strong>{result['prediction']}</strong> with
                    <strong>{result['confidence']}%</strong> confidence.
                </div>
                """, unsafe_allow_html=True)

                # Bonus Chart — FIXED: attendance set to 85% para may curve
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">📈 Study Hours vs Predicted Class</div>', unsafe_allow_html=True)

                with st.spinner("Building chart..."):
                    hours_range = np.linspace(0, 12, 25)
                    chart_rows = []
                    for h in hours_range:
                        try:
                            # I-vary ang attendance based sa study hours (para realistic)
                            varied_attendance = min(100, 50 + (h * 5))
                            r = requests.post(API_URL, json={
                                "study_hours": float(h),
                                "attendance_percentage": varied_attendance
                            }, timeout=5)
                            cls = 1 if r.json()["prediction"] == "Pass" else 0
                        except Exception:
                            cls = 0
                        chart_rows.append({
                            "Study Hours": round(float(h), 1),
                            "Predicted Class": cls
                        })
                    chart_df = pd.DataFrame(chart_rows).set_index("Study Hours")

                st.line_chart(chart_df, height=320, color="#a5b4fc")
                st.caption(
                    f"📍 Your input: **{study_hours} hours** and "
                    f"**{attendance}% attendance** → "
                    f"Predicted Class = **{result['prediction']}**"
                )

            else:
                st.error(f"❌ API Error {response.status_code}: {response.json()}")

        except requests.exceptions.ConnectionError:
            st.error("🚫 **Cannot connect to Flask API.**\n\nMake sure `python app.py` is running on port 5000 in another terminal.")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")

else:
    st.markdown("""
    <div class="placeholder">
        <span class="placeholder-emoji">🎯</span>
        <p>Fill in the student information above and click<br>
        <strong>🔮 Predict Result</strong> to see the prediction.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <strong>IT Elective 3 — AI: Machine Learning</strong><br>
    Model Persistence & Deployment<br>
    Frontend: Streamlit · Backend: Flask API · Model: Logistic Regression
</div>
""", unsafe_allow_html=True)

