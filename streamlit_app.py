import streamlit as st
import requests
import pandas as pd
import numpy as np

API_URL = "http://127.0.0.1:5000/predict"
DEFAULT_ATTENDANCE = 85.0

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background: linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdfa 100%); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 950px; }
    .hero {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 50%, #14b8a6 100%);
        padding: 3rem 2rem; border-radius: 24px; color: white;
        text-align: center; box-shadow: 0 20px 45px rgba(14, 165, 233, 0.35);
        margin-bottom: 2rem;
    }
    .hero-emoji { font-size: 4rem; display: block; margin-bottom: 0.5rem; }
    .hero h1 { font-size: 2.5rem; font-weight: 900; margin: 0; color: white !important; letter-spacing: -1px; }
    .hero p { font-size: 1.1rem; margin-top: 0.75rem; color: white !important; }
    .hero .badge {
        display: inline-block; background: rgba(255,255,255,0.28);
        padding: 0.4rem 1.1rem; border-radius: 999px; font-size: 0.85rem;
        margin-top: 1.2rem; color: white !important; font-weight: 700;
    }
    .section-title {
        font-size: 1.15rem; font-weight: 800; color: #0369a1;
        margin: 0 0 1.2rem 0; text-transform: uppercase; letter-spacing: 0.8px;
    }
    .stTextInput label, .stNumberInput label {
        font-weight: 700 !important; color: #0f172a !important; font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 12px !important; border: 2px solid #bae6fd !important;
        padding: 0.7rem 1rem !important; font-size: 1rem !important;
        background: #ffffff !important; color: #0f172a !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        color: white !important; border: none; padding: 0.85rem 2rem;
        border-radius: 12px; font-weight: 800; font-size: 1.05rem;
        width: 100%; box-shadow: 0 8px 22px rgba(14, 165, 233, 0.4);
    }
    .result-pass {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        padding: 2rem; border-radius: 20px; color: white !important;
        text-align: center; box-shadow: 0 15px 40px rgba(16, 185, 129, 0.35);
        margin-bottom: 1.5rem;
    }
    .result-fail {
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        padding: 2rem; border-radius: 20px; color: white !important;
        text-align: center; box-shadow: 0 15px 40px rgba(239, 68, 68, 0.35);
        margin-bottom: 1.5rem;
    }
    .result-emoji { font-size: 3.5rem; display: block; margin-bottom: 0.4rem; }
    .result-pass h2, .result-fail h2 {
        margin: 0; font-size: 1.9rem; font-weight: 900; color: white !important;
    }
    .result-pass p, .result-fail p {
        margin: 0.6rem 0 0 0; font-size: 1rem; color: white !important;
    }
    .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: white; padding: 1.4rem 1rem; border-radius: 16px;
        text-align: center; box-shadow: 0 4px 20px rgba(14, 165, 233, 0.12);
        border: 2px solid #e0f2fe;
    }
    .metric-label {
        font-size: 0.75rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 1.3px; font-weight: 800;
    }
    .metric-value { font-size: 1.8rem; font-weight: 900; color: #0369a1; margin-top: 0.4rem; }
    .info-box {
        background: white; border-left: 5px solid #0ea5e9;
        padding: 1.3rem 1.5rem; border-radius: 14px; margin-top: 1rem;
        color: #334155; font-size: 0.95rem; line-height: 1.7;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.1);
    }
    .info-box strong { color: #0369a1; }
    .info-box code {
        background: #f0f9ff; padding: 0.15rem 0.45rem; border-radius: 5px;
        color: #db2777; font-size: 0.85rem; border: 1px solid #bae6fd;
    }
    .placeholder {
        background: white; padding: 3.5rem 2rem; border-radius: 20px;
        border: 2px dashed #7dd3fc; text-align: center; color: #64748b;
    }
    .placeholder-emoji { font-size: 3.5rem; display: block; margin-bottom: 0.8rem; opacity: 0.7; }
    .placeholder p { font-size: 1.05rem; margin: 0; color: #475569; }
    .placeholder strong { color: #0369a1; }
    .footer {
        text-align: center; color: #64748b; font-size: 0.85rem;
        margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #cbd5e1;
    }
    .footer strong { color: #0369a1; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <span class="hero-emoji">🎓</span>
    <h1>Student Pass Predictor</h1>
    <p>AI-powered prediction using Machine Learning</p>
    <div class="badge">⚡ Flask API + Logistic Regression</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">📝 Student Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    student_name = st.text_input(
        "👤 Student Name",
        placeholder="e.g. Juan Dela Cruz",
        key="name_input"
    )
with col2:
    study_hours = st.number_input(
        "📚 Study Hours per Day",
        min_value=0.0, max_value=24.0, value=5.0, step=0.5
    )

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Predict Result", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

if predict_clicked:
    if not student_name.strip():
        st.warning("⚠️ Please enter the student's name.")
    else:
        payload = {
            "student_name": student_name,
            "study_hours": study_hours,
            "attendance_percentage": DEFAULT_ATTENDANCE
        }
        try:
            with st.spinner("🔄 Contacting Flask API..."):
                response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()

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

                st.markdown(f"""
                <div class="info-box">
                    <strong>🧠 How this was generated:</strong><br>
                    Streamlit sent a POST request to the Flask API.
                    The backend loaded <code>student_pass_model.joblib</code> and
                    made a prediction using <strong>{result['study_hours']} study hours</strong>.
                    The model returned <strong>{result['prediction']}</strong> with
                    <strong>{result['confidence']}%</strong> confidence.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">📈 Study Hours vs Predicted Class</div>', unsafe_allow_html=True)

                with st.spinner("Building chart..."):
                    hours_range = np.linspace(0, 12, 25)
                    chart_rows = []
                    for h in hours_range:
                        try:
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

                st.line_chart(chart_df, height=320, color="#0ea5e9")
                st.caption(
                    f"📍 Your input: **{study_hours} hours** → "
                    f"Predicted Class = **{result['prediction']}**"
                )
            else:
                st.error(f"❌ API Error {response.status_code}: {response.json()}")
        except requests.exceptions.ConnectionError:
            st.error("🚫 **Cannot connect to Flask API.**\n\nMake sure `python app.py` is running on port 5000.")
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

st.markdown("""
<div class="footer">
    <strong>IT Elective 3 — AI: Machine Learning</strong><br>
    Model Persistence & Deployment<br>
    Frontend: Streamlit · Backend: Flask API · Model: Logistic Regression
</div>
""", unsafe_allow_html=True)
