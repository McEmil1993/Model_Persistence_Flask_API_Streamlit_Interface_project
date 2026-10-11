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
        --card: rgba(255, 255, 255, 0.92);
        --panel: #ffffff;
        --primary: #4f46e5;
        --primary-soft: #eef2ff;
        --success: #10b981;
        --danger: #ef4444;
        --text: #1f2937;
        --muted: #6b7280;
        --line: #e5e7eb;
    }

    .stApp {
        background: linear-gradient(135deg, #eef4ff 0%, #f7f9fc 100%);
        color: var(--text);
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        background: linear-gradient(135deg, #1d4ed8 0%, #4f46e5 100%);
        border-radius: 22px;
        padding: 2rem 2.2rem;
        box-shadow: 0 18px 40px rgba(79, 70, 229, 0.18);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white !important;
        margin: 0;
        font-size: 2.3rem;
        font-weight: 700;
    }

    .hero p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.6rem 0 0 0;
        font-size: 1.05rem;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .metric-box {
        background: linear-gradient(180deg, #ffffff 0%, #f8faff 100%);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.15rem;
        text-align: center;
    }

    .metric-box h3 {
        margin: 0;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        color: var(--muted);
        text-transform: uppercase;
    }

    .metric-box .value {
        display: block;
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--text);
        margin-top: 0.35rem;
    }

    .result-pass {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        color: #065f46;
    }

    .result-fail {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        color: #991b1b;
    }

    .result-pass h3,
    .result-fail h3 {
        margin: 0 0 0.45rem 0;
        font-size: 1.1rem;
    }

    .result-pass p,
    .result-fail p {
        margin: 0;
        font-size: 1rem;
    }

    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        height: 3.1rem;
        width: 100%;
    }

    [data-testid="stFormSubmitButton"] > button:hover {
        filter: brightness(1.04);
    }

    .stSlider > div {
        padding-top: 0.5rem;
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
    <div class="hero">
        <h1>🎓 Student Pass Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        col_input, col_side = st.columns([1.6, 0.9])

        with col_input:
            student_name = st.text_input("Student name", placeholder="Enter student name")
            study_hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
            attendance_percentage = st.slider("Attendance percentage", min_value=0, max_value=100, value=75)

        with col_side:
            st.markdown(
                """
                <div class="metric-box">
                    <h3>Model</h3>
                    <span class="value">Logistic Regression</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="metric-box">
                    <h3>Based on</h3>
                    <span class="value">Study + Attendance</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button("Predict outcome", use_container_width=True)

    if submitted:
        if not student_name.strip():
            st.warning("Please enter the student's name first.")
        else:
            sample = pd.DataFrame({
                "study_hours": [study_hours],
                "attendance_percentage": [attendance_percentage],
            })
            prediction = int(model.predict(sample)[0])
            probability = float(model.predict_proba(sample)[0][1])

            result_class = "result-pass" if prediction == 1 else "result-fail"
            status_label = "PASS" if prediction == 1 else "FAIL"
            status_icon = "✅" if prediction == 1 else "❌"
            accent = "#10b981" if prediction == 1 else "#ef4444"

            st.markdown(
                f"""
                <div class="{result_class}">
                    <h3>{status_icon} {student_name} predicted to {status_label}</h3>
                    <p>Pass probability: <strong>{probability:.0%}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_cols = st.columns(3)
            metric_cols[0].metric("Study hours", f"{study_hours:.1f} hrs")
            metric_cols[1].metric("Attendance", f"{attendance_percentage}%")
            metric_cols[2].metric("Probability", f"{probability:.0%}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(probability)
            st.caption(f"Decision threshold: 50% | Model confidence: {probability:.0%}")

            coef_hours = float(model.coef_[0][0])
            coef_attendance = float(model.coef_[0][1])
            intercept = float(model.intercept_[0])

            with st.expander("How this prediction was generated"):
                st.write(
                    f"The saved model is a logistic regression trained on study hours and attendance percentage. "
                    f"It loaded its learned values and applied them to the {study_hours} hours and {attendance_percentage}% attendance you entered."
                )
                st.latex(r"P(\text{pass}) = \frac{1}{1 + e^{-(b_0 + b_1 \cdot \text{hours} + b_2 \cdot \text{attendance})}}")
                st.write(
                    f"With the learned values b₀ = {intercept:.2f}, b₁ = {coef_hours:.2f}, and b₂ = {coef_attendance:.2f}, "
                    f"the pass probability is {probability:.2f}. If it is above 0.5, the model predicts PASS; otherwise it predicts FAIL."
                )

            st.subheader("Study hours vs predicted class")
            hours_range = np.linspace(0, 12, 200)
            curve = model.predict_proba(
                pd.DataFrame({
                    "study_hours": hours_range,
                    "attendance_percentage": [attendance_percentage] * len(hours_range),
                })
            )[:, 1]

            fig, ax = plt.subplots(figsize=(8, 4.6), facecolor="#f8fbff")
            ax.plot(hours_range, curve, color="#4f46e5", linewidth=2.5, label="Pass probability curve")
            ax.axhline(0.5, color="#64748b", linestyle="--", linewidth=1.2, label="PASS / FAIL threshold (0.5)")
            ax.scatter(
                [study_hours],
                [probability],
                color=accent,
                s=120,
                zorder=5,
                label=f"{student_name}: {status_label}",
            )
            ax.set_xlabel("Study hours")
            ax.set_ylabel("Pass probability")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title("Probability trend across study hours")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
            fig.tight_layout()
            st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)