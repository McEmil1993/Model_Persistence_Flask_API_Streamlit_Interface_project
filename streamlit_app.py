from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model_v2.joblib"

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #f4f7fb;
            --panel: rgba(255, 255, 255, 0.9);
            --border: rgba(148, 163, 184, 0.22);
            --primary: #4f46e5;
            --primary-soft: #eef2ff;
            --success: #16a34a;
            --danger: #dc2626;
            --text: #0f172a;
            --muted: #475569;
            --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 2.5rem;
            padding-bottom: 2rem;
        }

        .app-header {
            margin-bottom: 1.5rem;
            padding: 0.5rem 0 0.2rem 0;
        }

        .app-header h1 {
            margin: 0;
            font-size: 2.7rem;
            font-weight: 800;
            letter-spacing: -0.06em;
            color: #0f172a;
        }

        .app-header p {
            margin-top: 0.5rem;
            font-size: 1.05rem;
            color: var(--muted);
        }

        .glass-panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 1.25rem 1.3rem;
            backdrop-filter: blur(8px);
        }

        .info-card {
            background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
            border: 1px solid rgba(79, 70, 229, 0.12);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            height: 100%;
        }

        .info-card h3 {
            margin: 0 0 0.8rem 0;
            color: var(--text);
            font-size: 1.1rem;
        }

        .info-card p {
            margin: 0.2rem 0;
            font-size: 0.96rem;
            color: var(--muted);
            line-height: 1.6;
        }

        .result-box {
            border-radius: 18px;
            padding: 1.2rem 1.25rem;
            margin-top: 1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: #f8fafc;
        }

        .result-box.pass {
            background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
            border-color: rgba(22, 163, 74, 0.3);
        }

        .result-box.fail {
            background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
            border-color: rgba(220, 38, 38, 0.25);
        }

        .result-box h3 {
            margin: 0 0 0.4rem 0;
            font-size: 1.3rem;
        }

        .result-box p {
            margin: 0;
            font-size: 1rem;
            color: var(--text);
        }

        div[data-testid="stForm"] {
            background: transparent;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, var(--primary) 0%, #7c3aed 100%);
            color: white;
            font-weight: 700;
            padding: 0.8rem 1rem;
            box-shadow: 0 12px 28px rgba(79, 70, 229, 0.25);
        }

        div.stButton > button:hover {
            filter: brightness(1.03);
        }

        .stSlider > div[data-baseweb="slider"] {
            margin-top: 0.5rem;
        }

        .stMetric {
            background: rgba(255, 255, 255, 0.65);
            border-radius: 14px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            padding: 0.7rem 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}. Run 'py train_model_v2.py' first.")
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

st.markdown(
    """
    <div class="app-header">
        <h1>🎓 Student Pass Predictor</h1>
        <p>Estimate whether a student is likely to pass based on study effort and attendance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.25, 0.75])

with left_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    with st.form("prediction_form", clear_on_submit=False):
        student_name = st.text_input("Student name", placeholder="e.g. Maria")
        study_hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        attendance_percentage = st.slider("Attendance percentage", min_value=0, max_value=100, value=75)
        submitted = st.form_submit_button("Predict pass likelihood")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown(
        """
        <div class="info-card">
            <h3>Model snapshot</h3>
            <p><strong>Type:</strong> Logistic regression</p>
            <p><strong>Features:</strong> Study hours, attendance</p>
            <p><strong>Output:</strong> Pass or fail prediction</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is not None:
        coef_hours = float(model.coef_[0][0])
        coef_attendance = float(model.coef_[0][1])
        intercept = float(model.intercept_[0])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Intercept", f"{intercept:.2f}")
        with c2:
            st.metric("Hours coef.", f"{coef_hours:.2f}")

        st.caption(f"Attendance coefficient: {coef_attendance:.2f}")

if submitted:
    if model is None:
        st.stop()
    if not student_name.strip():
        st.warning("Please enter the student's name first.")
    else:
        sample = pd.DataFrame({
            "study_hours": [study_hours],
            "attendance_percentage": [attendance_percentage],
        })
        prediction = int(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0][1])

        status_class = "pass" if prediction == 1 else "fail"
        status_text = "PASS" if prediction == 1 else "FAIL"
        emoji = "✅" if prediction == 1 else "❌"

        st.markdown(
            f'''
            <div class="result-box {status_class}">
                <h3>{emoji} {student_name} is predicted to {status_text}</h3>
                <p>Confidence: {probability:.0%}</p>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        coef_hours = float(model.coef_[0][0])
        coef_attendance = float(model.coef_[0][1])
        intercept = float(model.intercept_[0])

        st.subheader("How the prediction is generated")
        st.write(
            f"The model combines the entered study hours ({study_hours}) and attendance ({attendance_percentage}%) into a logistic regression formula."
        )
        st.latex(r"P(\text{pass}) = \frac{1}{1 + e^{-(b_0 + b_1 \cdot \text{hours} + b_2 \cdot \text{attendance})}}")
        st.write(
            f"With learned values b₀ = {intercept:.2f}, b₁ = {coef_hours:.2f}, and b₂ = {coef_attendance:.2f}, the computed pass probability is {probability:.2f}."
        )

        st.subheader("Probability curve")
        hours_range = np.linspace(0, 12, 200)
        curve = model.predict_proba(
            pd.DataFrame({
                "study_hours": hours_range,
                "attendance_percentage": [attendance_percentage] * len(hours_range),
            })
        )[:, 1]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(hours_range, curve, color="#4f46e5", linewidth=2.5, label="Pass probability curve")
        ax.axhline(0.5, color="#64748b", linestyle="--", linewidth=1.2, label="Threshold (0.5)")
        ax.scatter(
            [study_hours],
            [probability],
            color="#16a34a" if prediction == 1 else "#dc2626",
            s=120,
            zorder=5,
            label=f"{student_name}: {status_text}",
        )
        ax.set_xlabel("Study hours")
        ax.set_ylabel("Pass probability")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False)
        ax.set_title("Model probability by study hours")
        fig.tight_layout()
        st.pyplot(fig)