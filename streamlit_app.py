"""Student Pass/Fail Studio -- Redesigned High-Performance UI.

An interactive machine learning dashboard for scoring student success probability,
analyzing sensitivity curves, exploring decision boundaries, and inspecting the model pipeline.
"""

from pathlib import Path
import warnings

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Suppress minor unpickling version discrepancy notices
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pass/Fail Studio • ML Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Model Loading & Analytical Inversion
# ---------------------------------------------------------
MODEL_FILE = Path(__file__).with_name("student_pass_model.joblib")


@st.cache_resource(show_spinner=False)
def load_trained_pipeline():
    """Load the persisted scikit-learn pipeline (StandardScaler + LogisticRegression)."""
    return joblib.load(MODEL_FILE)


clf = load_trained_pipeline()
scaler = clf.named_steps.get("standardscaler")
logistic_reg = clf.named_steps.get("logisticregression")

mu_h, mu_a = float(scaler.mean_[0]), float(scaler.mean_[1])
scale_h, scale_a = float(scaler.scale_[0]), float(scaler.scale_[1])
w_h, w_a = float(logistic_reg.coef_[0][0]), float(logistic_reg.coef_[0][1])
intercept = float(logistic_reg.intercept_[0])


def calculate_pass_threshold_hours(attendance_pct: float) -> float:
    """Calculate the exact minimum study hours needed to achieve P(pass) >= 0.50."""
    z_a = (attendance_pct - mu_a) / scale_a
    z_h = -(intercept + w_a * z_a) / w_h
    req_hours = mu_h + z_h * scale_h
    return max(0.0, float(req_hours))


# ---------------------------------------------------------
# Custom Modern Design System & CSS
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Header styling */
.hero-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.5);
}

.hero-header::after {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.3) 0%, transparent 70%);
    pointer-events: none;
    border-radius: 50%;
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #A5B4FC;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.live-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10B981;
    box-shadow: 0 0 10px #10B981;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0;
    line-height: 1.2;
}

.header-subtitle {
    color: #94A3B8;
    font-size: 0.98rem;
    margin-top: 8px;
    max-width: 680px;
    line-height: 1.5;
}

/* Verdict Display */
.verdict-banner-pass {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(6, 95, 70, 0.25) 100%);
    border: 1.5px solid rgba(52, 211, 153, 0.45);
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15);
    margin-bottom: 16px;
}

.verdict-banner-fail {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.45) 0%, rgba(153, 27, 27, 0.25) 100%);
    border: 1.5px solid rgba(248, 113, 113, 0.45);
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.15);
    margin-bottom: 16px;
}

.verdict-label {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #CBD5E1;
    margin-bottom: 6px;
}

.verdict-text-pass {
    font-size: 3.2rem;
    font-weight: 900;
    color: #34D399;
    text-shadow: 0 0 24px rgba(52, 211, 153, 0.5);
    line-height: 1.1;
    margin-bottom: 6px;
}

.verdict-text-fail {
    font-size: 3.2rem;
    font-weight: 900;
    color: #F87171;
    text-shadow: 0 0 24px rgba(248, 113, 113, 0.5);
    line-height: 1.1;
    margin-bottom: 6px;
}

.metric-pill-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 18px;
}

.metric-pill {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 12px 14px;
    text-align: center;
}

.metric-pill-label {
    font-size: 0.72rem;
    color: #94A3B8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.metric-pill-value {
    font-size: 1.25rem;
    color: #F8FAFC;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

/* Pill badges */
.tech-pill {
    display: inline-block;
    padding: 4px 10px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: #E2E8F0;
    margin-right: 6px;
    margin-bottom: 6px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization for Interactive Presets
# ---------------------------------------------------------
if "student_name" not in st.session_state:
    st.session_state.student_name = "Alex Rivera"
if "study_hours" not in st.session_state:
    st.session_state.study_hours = 4.50
if "attendance_pct" not in st.session_state:
    st.session_state.attendance_pct = 80.0

# ---------------------------------------------------------
# Sidebar: Controls & Scenarios
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Studio Controls")
    st.caption("Quickly test scenarios or adjust input parameters.")

    st.markdown("#### Preset Student Personas")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("👤 Typical", use_container_width=True, help="4.5h study, 80% attendance"):
            st.session_state.student_name = "Jordan Lee"
            st.session_state.study_hours = 4.50
            st.session_state.attendance_pct = 80.0
            st.rerun()

        if st.button("⚠️ Borderline", use_container_width=True, help="1.6h study, 70% attendance"):
            st.session_state.student_name = "Taylor Casey"
            st.session_state.study_hours = 1.60
            st.session_state.attendance_pct = 70.0
            st.rerun()

    with col_p2:
        if st.button("⭐ Honors", use_container_width=True, help="9.0h study, 95% attendance"):
            st.session_state.student_name = "Morgan Chen"
            st.session_state.study_hours = 9.00
            st.session_state.attendance_pct = 95.0
            st.rerun()

        if st.button("🚨 At Risk", use_container_width=True, help="1.0h study, 45% attendance"):
            st.session_state.student_name = "Riley Smith"
            st.session_state.study_hours = 1.00
            st.session_state.attendance_pct = 45.0
            st.rerun()

    st.divider()

    st.markdown("#### Input Profile")
    student_name = st.text_input(
        "Student Name",
        value=st.session_state.student_name,
        placeholder="e.g. Sam Morgan",
        help="Optional name label used in reports.",
    )
    st.session_state.student_name = student_name

    hours = st.slider(
        "Study Hours / Week",
        min_value=0.0,
        max_value=12.0,
        value=float(st.session_state.study_hours),
        step=0.25,
        format="%.2f hrs",
        help="Weekly hours dedicated to self-study outside class lectures.",
    )
    st.session_state.study_hours = hours

    attendance = st.slider(
        "Attendance Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.attendance_pct),
        step=1.0,
        format="%.0f%%",
        help="Percentage of lectures and lab sessions attended.",
    )
    st.session_state.attendance_pct = attendance

    st.divider()
    st.markdown("#### Pipeline Artifacts")
    st.caption("StandardScaler + LogisticRegression (persisted via joblib).")

# ---------------------------------------------------------
# Core Model Prediction Calculation
# ---------------------------------------------------------
prob_pass = float(clf.predict_proba([[hours, attendance]])[0][1])
is_pass = prob_pass >= 0.50
verdict_str = "PASS" if is_pass else "FAIL"
display_name = student_name.strip() or "Student"
req_hours = calculate_pass_threshold_hours(attendance)
hour_diff = hours - req_hours

# ---------------------------------------------------------
# App Header Hero Section
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-header">
        <div class="brand-badge">
            <span class="live-indicator"></span>
            Machine Learning • Logistic Inference Studio
        </div>
        <h1 class="header-title">Student Pass / Fail Predictor</h1>
        <div class="header-subtitle">
            Evaluate academic success probabilities with standard-scaled logistic regression.
            Explore interactive sensitivity sweeps, what-if margins, and multi-dimensional decision contours.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Studio Tabs
# ---------------------------------------------------------
tab_prediction, tab_sensitivity, tab_landscape, tab_model = st.tabs(
    [
        "🎯 Inference Dashboard",
        "📈 Sensitivity Curve",
        "🗺️ 2D Decision Landscape",
        "🧠 Model & Architecture",
    ]
)

# ---------------------------------------------------------
# TAB 1: Main Inference Dashboard
# ---------------------------------------------------------
with tab_prediction:
    col_input_summary, col_verdict_card = st.columns([1, 1.2], gap="large")

    with col_input_summary:
        with st.container(border=True):
            st.markdown("### 📝 Student Profile")
            st.write(f"Target Candidate: **{display_name}**")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Study Time", f"{hours:.2f} hrs/wk")
            with col_m2:
                st.metric("Attendance", f"{attendance:.0f}%")

            st.divider()
            st.markdown("#### 🎯 Passing Threshold Calibration")
            if is_pass:
                st.success(
                    f"**Safety Margin:** At {attendance:.0f}% attendance, the minimum study threshold "
                    f"to pass is **{req_hours:.2f} hrs/wk**. {display_name} is currently **{hour_diff:+.2f} hrs** above threshold."
                )
            else:
                hours_needed = max(0.0, req_hours - hours)
                st.error(
                    f"**Path to Pass:** At {attendance:.0f}% attendance, {display_name} needs at least "
                    f"**{req_hours:.2f} hrs/wk** (+{hours_needed:.2f} additional hours) to cross the 50% pass boundary."
                )

    with col_verdict_card:
        banner_class = "verdict-banner-pass" if is_pass else "verdict-banner-fail"
        text_class = "verdict-text-pass" if is_pass else "verdict-text-fail"
        badge_symbol = "✅" if is_pass else "⚠️"

        st.markdown(
            f"""
            <div class="{banner_class}">
                <div class="verdict-label">Predicted Outcome for {display_name}</div>
                <div class="{text_class}">{badge_symbol} {verdict_str}</div>
                <div style="font-size: 1.15rem; color: #CBD5E1; margin-bottom: 14px;">
                    Success Probability: <b>{prob_pass * 100:.1f}%</b>
                </div>
                <div class="metric-pill-grid">
                    <div class="metric-pill">
                        <div class="metric-pill-label">P(Pass)</div>
                        <div class="metric-pill-value">{prob_pass:.4f}</div>
                    </div>
                    <div class="metric-pill">
                        <div class="metric-pill-label">Cutoff</div>
                        <div class="metric-pill-value">0.5000</div>
                    </div>
                    <div class="metric-pill">
                        <div class="metric-pill-label">Confidence</div>
                        <div class="metric-pill-value">{"High" if abs(prob_pass - 0.5) >= 0.25 else "Moderate"}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(prob_pass, 0.0), 1.0))

    # Automated Advisory Section
    st.divider()
    with st.container(border=True):
        st.markdown("### 📋 Automated Academic Advisory")
        if is_pass:
            st.markdown(
                f"""
                > **Positive Outlook:** With **{hours:.2f} hours** of weekly study and **{attendance:.0f}%** class attendance,
                > `{display_name}` shows solid performance indicators with a **{prob_pass * 100:.1f}%** pass probability.
                > Maintaining this momentum will keep the student securely in the passing zone.
                """
            )
        else:
            st.markdown(
                f"""
                > ⚠️ **Intervention Recommended:** With **{hours:.2f} hours** of study and **{attendance:.0f}%** attendance,
                > `{display_name}` currently falls below the model's 50% pass threshold (estimated probability: **{prob_pass * 100:.1f}%**).
                > Dedicating at least **{req_hours:.2f} hours/week** to study (+{req_hours - hours:.2f} hrs) will elevate them to passing status.
                """
            )

# ---------------------------------------------------------
# TAB 2: Interactive Probability & Sensitivity Curve
# ---------------------------------------------------------
with tab_sensitivity:
    with st.container(border=True):
        st.markdown("### 📈 Study Hours Sensitivity Curve")
        st.write(
            f"Shows the predicted probability of passing across the entire 0 – 12 hours spectrum, "
            f"with attendance fixed at **{attendance:.0f}%**. The marker indicates **{display_name}'s** current position."
        )

        # Generate fine grid for smooth curve
        sweep_hours = np.linspace(0.0, 12.0, 120)
        sweep_probs = [
            float(clf.predict_proba([[h, attendance]])[0][1]) for h in sweep_hours
        ]
        sweep_df = pd.DataFrame(
            {
                "Study Hours": sweep_hours,
                "Pass Probability": sweep_probs,
            }
        )

        current_point_df = pd.DataFrame(
            {
                "Study Hours": [hours],
                "Pass Probability": [prob_pass],
                "Label": [f"{display_name} ({hours:.2f}h, {prob_pass * 100:.1f}%)"],
            }
        )

        # Base curve
        base_curve = (
            alt.Chart(sweep_df)
            .mark_line(color="#6366F1", strokeWidth=3.5)
            .encode(
                x=alt.X("Study Hours:Q", title="Weekly Study Hours (hrs)"),
                y=alt.Y(
                    "Pass Probability:Q",
                    title="Probability P(Pass)",
                    scale=alt.Scale(domain=[0.0, 1.0]),
                ),
                tooltip=[
                    alt.Tooltip("Study Hours:Q", format=".2f"),
                    alt.Tooltip("Pass Probability:Q", format=".2%"),
                ],
            )
        )

        # Threshold 0.5 reference line
        threshold_line = (
            alt.Chart(pd.DataFrame({"y": [0.5]}))
            .mark_rule(color="#EF4444", strokeDash=[6, 4], strokeWidth=2)
            .encode(y="y:Q")
        )

        # Student marker
        student_marker = (
            alt.Chart(current_point_df)
            .mark_point(
                color="#34D399" if is_pass else "#F87171",
                size=220,
                filled=True,
                stroke="white",
                strokeWidth=2,
            )
            .encode(
                x="Study Hours:Q",
                y="Pass Probability:Q",
                tooltip=[
                    alt.Tooltip("Label:N", title="Student"),
                    alt.Tooltip("Study Hours:Q", format=".2f"),
                    alt.Tooltip("Pass Probability:Q", format=".2%"),
                ],
            )
        )

        chart = (
            (base_curve + threshold_line + student_marker)
            .properties(height=380)
            .configure_view(stroke="transparent")
        )

        st.altair_chart(chart, use_container_width=True)

        with st.expander("🔍 View Simulation Data Table", expanded=False):
            sample_hours = [round(x * 0.5, 1) for x in range(0, 25)]
            sample_probs = [
                float(clf.predict_proba([[h, attendance]])[0][1]) for h in sample_hours
            ]
            sample_classes = ["PASS" if p >= 0.5 else "FAIL" for p in sample_probs]
            tab_df = pd.DataFrame(
                {
                    "Study Hours (hrs)": sample_hours,
                    "Probability": [round(p, 4) for p in sample_probs],
                    "Success Chance": [f"{p * 100:.1f}%" for p in sample_probs],
                    "Outcome": sample_classes,
                }
            )
            st.dataframe(tab_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 3: 2D Decision Boundary Landscape
# ---------------------------------------------------------
with tab_landscape:
    with st.container(border=True):
        st.markdown("### 🗺️ Feature Plane Decision Landscape")
        st.write(
            "Visualizes how Study Hours and Attendance jointly determine classification. "
            "The red dashed line marks the mathematical boundary where $P(\\text{pass}) = 0.50$."
        )

        # Compute boundary line: for attendance 0 to 100, calculate boundary hours
        boundary_att = np.linspace(0.0, 100.0, 50)
        boundary_h = [calculate_pass_threshold_hours(a) for a in boundary_att]
        boundary_df = pd.DataFrame(
            {
                "Attendance": boundary_att,
                "Study Hours": boundary_h,
            }
        )

        student_pos_df = pd.DataFrame(
            {
                "Attendance": [attendance],
                "Study Hours": [hours],
                "Status": ["PASS" if is_pass else "FAIL"],
            }
        )

        boundary_chart = (
            alt.Chart(boundary_df)
            .mark_line(color="#EF4444", strokeDash=[6, 4], strokeWidth=3)
            .encode(
                x=alt.X("Study Hours:Q", title="Study Hours", scale=alt.Scale(domain=[0, 12])),
                y=alt.Y(
                    "Attendance:Q",
                    title="Attendance Percentage (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=[
                    alt.Tooltip("Study Hours:Q", format=".2f"),
                    alt.Tooltip("Attendance:Q", format=".0f"),
                ],
            )
        )

        point_chart = (
            alt.Chart(student_pos_df)
            .mark_point(
                color="#34D399" if is_pass else "#F87171",
                size=260,
                filled=True,
                stroke="white",
                strokeWidth=2.5,
            )
            .encode(
                x="Study Hours:Q",
                y="Attendance:Q",
                tooltip=[
                    alt.Tooltip("Study Hours:Q", format=".2f"),
                    alt.Tooltip("Attendance:Q", format=".0f"),
                    alt.Tooltip("Status:N", title="Verdict"),
                ],
            )
        )

        boundary_combined = (
            (boundary_chart + point_chart)
            .properties(height=380)
            .configure_view(stroke="transparent")
        )

        st.altair_chart(boundary_combined, use_container_width=True)

        st.info(
            f"💡 **Model Trade-off:** With current weights, **1.0 hour of study** compensates for "
            f"approx. **{abs((w_h / scale_h) / (w_a / scale_a)):.1f}% attendance**. "
            f"Higher attendance allows a student to safely pass with fewer hours of study."
        )

# ---------------------------------------------------------
# TAB 4: Model Diagnostics & REST API Guide
# ---------------------------------------------------------
with tab_model:
    col_pipeline, col_api = st.columns(2, gap="large")

    with col_pipeline:
        with st.container(border=True):
            st.markdown("### ⚙️ Pipeline Composition")
            st.markdown(
                """
                <div style="margin-bottom: 12px;">
                    <span class="tech-pill">StandardScaler</span>
                    <span class="tech-pill">LogisticRegression(max_iter=1000)</span>
                    <span class="tech-pill">Joblib Persistence</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(
                "Because Study Hours (0–10) and Attendance (0–100) operate on different scales, "
                "standardizing both features prevents attendance from dominating numerical optimization."
            )

            st.markdown("##### 📐 Learned Parameters")
            st.code(
                f"StandardScaler Mean:    hours={mu_h:.4f}, attendance={mu_a:.4f}\n"
                f"StandardScaler Scale:   hours={scale_h:.4f}, attendance={scale_a:.4f}\n"
                f"Logistic Weights (W):   w_hours={w_h:.4f}, w_attendance={w_a:.4f}\n"
                f"Logistic Intercept (B): {intercept:.4f}",
                language="python",
            )

    with col_api:
        with st.container(border=True):
            st.markdown("### 🌐 Companion REST API (Flask)")
            st.write(
                "The accompanying Flask microservice (`app.py`) serves the identical model pipeline over HTTP POST:"
            )

            st.markdown("##### PowerShell Call")
            st.code(
                f'Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post \\\n'
                f'  -ContentType "application/json" \\\n'
                f'  -Body \'{{"study_hours": {hours}, "attendance_percentage": {attendance}}}\'',
                language="powershell",
            )

            st.markdown("##### Expected Output")
            st.code(
                f'{{\n'
                f'  "study_hours": {hours},\n'
                f'  "attendance_percentage": {attendance},\n'
                f'  "prediction": {1 if is_pass else 0},\n'
                f'  "result": "{verdict_str}",\n'
                f'  "probability_pass": {prob_pass:.4f}\n'
                f'}}',
                language="json",
            )
