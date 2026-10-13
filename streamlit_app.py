
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Pass Predictor | Danger Mode",
    page_icon="🔴",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# RED DANGER THEME
# ============================================================

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at top, #3b1018 0%, #150b10 42%, #080808 100%);
        color: #f5f5f5;
    }

    /* Main content width and spacing */
    .block-container {
        max-width: 850px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Text colors */
    h1, h2, h3, p, label, .stMarkdown {
        color: #f5f5f5;
    }

    h1 {
        color: #ff3030 !important;
        font-weight: 900 !important;
        text-align: center;
        text-shadow: 0 0 18px rgba(255, 0, 0, 0.45);
    }

    /* Subtitle */
    .subtitle {
        color: #c7b7bc;
        text-align: center;
        font-size: 16px;
        margin-bottom: 28px;
    }

    /* Header badge */
    .danger-badge {
        background: rgba(255, 30, 50, 0.12);
        border: 1px solid #ff3030;
        border-radius: 30px;
        padding: 7px 16px;
        color: #ff4b5c;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 2px;
        text-align: center;
        width: fit-content;
        margin: 0 auto 18px auto;
    }

    /* Input fields */
    .stTextInput input,
    .stNumberInput input {
        background-color: #171114 !important;
        color: white !important;
        border: 1px solid #7f2634 !important;
        border-radius: 10px !important;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus {
        border: 1px solid #ff3030 !important;
        box-shadow: 0 0 10px rgba(255, 48, 48, 0.25) !important;
    }

    /* Primary button */
    .stButton > button {
        width: 100%;
        min-height: 48px;
        color: white !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        letter-spacing: 1px;
        background: linear-gradient(90deg, #b30018, #ff263f) !important;
        border: 1px solid #ff4b5c !important;
        border-radius: 10px !important;
        transition: 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #ff263f, #ff5267) !important;
        box-shadow: 0 0 20px rgba(255, 30, 50, 0.45);
        transform: translateY(-2px);
    }

    /* Section cards */
    .panel {
        background: rgba(28, 17, 22, 0.92);
        border: 1px solid #54202b;
        border-radius: 16px;
        padding: 22px;
        margin: 18px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    }

    /* Prediction result */
    .result-card {
        border-radius: 15px;
        padding: 24px;
        margin-top: 22px;
        text-align: center;
        background: #1a1014;
    }

    .pass-card {
        border: 1px solid #28a879;
        box-shadow: 0 0 20px rgba(40, 168, 121, 0.12);
    }

    .fail-card {
        border: 1px solid #ff263f;
        box-shadow: 0 0 22px rgba(255, 38, 63, 0.18);
    }

    .result-label {
        font-size: 13px;
        letter-spacing: 3px;
        color: #bbaab0;
        font-weight: bold;
    }

    .result-value {
        font-size: 38px;
        font-weight: 900;
        margin: 8px 0;
    }

    .pass-text {
        color: #36d399;
    }

    .fail-text {
        color: #ff4055;
        text-shadow: 0 0 15px rgba(255, 40, 60, 0.25);
    }

    .student-text {
        color: #ded2d6;
        font-size: 16px;
    }

    /* Divider */
    hr {
        border-color: #54202b !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Footer */
    .footer {
        color: #76676c;
        text-align: center;
        font-size: 12px;
        padding-top: 25px;
    }

    /* Chart container */
    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"],
    [data-testid="stElementContainer"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "Model file not found: student_pass_model.joblib. "
        "Please run train_model.py first."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load the saved model: {exc}")
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="danger-badge">● SYSTEM ONLINE · ML PREDICTION</div>',
    unsafe_allow_html=True,
)

st.title("🔴 STUDENT PASS PREDICTOR")

st.markdown(
    """
    <div class="subtitle">
        Academic performance prediction system<br>
        Enter the student's details to generate a prediction.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ============================================================
# INPUT FORM
# ============================================================

st.markdown("### ⚠️ STUDENT ASSESSMENT")

with st.container(border=True):
    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student's full name",
    )

    study_hours = st.number_input(
        "Daily Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5,
        help="Enter the number of hours the student studies per day.",
    )

    st.caption(
        "INPUT RANGE: 0–24 HOURS  |  "
        "PREDICTION ENGINE: SAVED MACHINE-LEARNING MODEL"
    )

    predict_clicked = st.button(
        "⚡ RUN PREDICTION",
        type="primary",
        use_container_width=True,
    )

# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:
    if not student_name.strip():
        st.warning("⚠️ Please enter the student's name before predicting.")
    else:
        try:
            # Use a DataFrame with the expected feature name.
            input_data = pd.DataFrame(
                {"study_hours": [study_hours]}
            )

            prediction = int(model.predict(input_data)[0])
            student = student_name.strip()

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-card pass-card">
                        <div class="result-label">PREDICTION RESULT</div>
                        <div class="result-value pass-text">
                            ✓ PASS
                        </div>
                        <div class="student-text">
                            {student.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}
                        </div>
                        <p style="color:#36d399;">
                            The model predicts that this student will pass.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif prediction == 0:
                st.markdown(
                    f"""
                    <div class="result-card fail-card">
                        <div class="result-label">PREDICTION RESULT</div>
                        <div class="result-value fail-text">
                            ✖ FAIL
                        </div>
                        <div class="student-text">
                            {student.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}
                        </div>
                        <p style="color:#ff6576;">
                            The model predicts that this student may fail.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning(
                    f"The model returned an unexpected prediction: {prediction}"
                )
                st.stop()

            # Model information
            st.markdown("### 📊 ASSESSMENT DETAILS")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Daily Study Hours", f"{study_hours:g} hrs")

            with col2:
                st.metric(
                    "Predicted Outcome",
                    "PASS" if prediction == 1 else "FAIL",
                )

            st.info(
                "This result was generated using the saved model "
                "`student_pass_model.joblib`. The model was loaded for "
                "prediction and was not retrained."
            )

            # Study-hours visualization
            st.markdown("### 📈 STUDY HOURS CHART")

            chart_data = pd.DataFrame(
                {
                    "Study Hours": [study_hours],
                    "Predicted Class": [prediction],
                }
            )

            st.bar_chart(
                chart_data.set_index("Study Hours"),
                color="#ff3030",
            )

            st.caption(
                "The chart shows the model's predicted class for this input. "
                "It does not show a pass probability or prove that study hours "
                "caused the outcome."
            )

        except Exception as exc:
            st.error(
                "Prediction failed. Check that the saved model accepts "
                "the 'study_hours' input feature and returns 0 for Fail "
                "or 1 for Pass."
            )
            st.exception(exc)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        STUDENT PASS PREDICTOR · DANGER THEME<br>
        MACHINE-LEARNING OUTPUT IS AN ESTIMATE, NOT A GUARANTEE.
    </div>
    """,
    unsafe_allow_html=True,
)
