"""Pass/Fail Studio -- Premium Modern Streamlit Interface.

Interactive diagnostic and predictive interface for student academic performance,
powered by a pre-trained scikit-learn logistic regression pipeline.
"""

from pathlib import Path
import warnings

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Page Configuration & Aesthetic Injection
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI • Pass/Fail Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling (Modern Glassmorphic Dark-Mode / High-Contrast Theme)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography & Base */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.02rem;
        margin: 0;
        max-width: 720px;
        line-height: 1.5;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Verdict Display Badges */
    .verdict-box-pass {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 1.5px solid rgba(16, 185, 129, 0.4);
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
    }

    .verdict-box-fail {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.08) 100%);
        border: 1.5px solid rgba(239, 68, 68, 0.4);
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.15);
    }

    .verdict-tag {
        display: inline-block;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin: 4px 0 10px 0;
    }
    .verdict-pass-text {
        color: #34D399;
        text-shadow: 0 0 20px rgba(52, 211, 153, 0.4);
    }
    .verdict-fail-text {
        color: #F87171;
        text-shadow: 0 0 20px rgba(248, 113, 113, 0.4);
    }

    /* Custom Meter */
    .prob-meter-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        height: 14px;
        width: 100%;
        overflow: hidden;
        margin: 14px 0 8px 0;
        position: relative;
    }
    .prob-meter-fill-pass {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #10B981 0%, #34D399 100%);
        box-shadow: 0 0 10px rgba(52, 211, 153, 0.5);
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .prob-meter-fill-fail {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #DC2626 0%, #F87171 100%);
        box-shadow: 0 0 10px rgba(248, 113, 113, 0.5);
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Metric Grid Cards */
    .stat-pill {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }
    .stat-pill-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .stat-pill-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Recommendation Banner */
    .rec-box {
        background: rgba(99, 102, 241, 0.08);
        border-left: 4px solid #6366F1;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin-top: 14px;
        color: #E2E8F0;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Button Tweaks */
    div.stButton > button:first-child {
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Model Loading & Cached Metadata
# -----------------------------------------------------------------------------
MODEL_FILE = Path(__file__).with_name("student_pass_model.joblib")


@st.cache_resource(show_spinner="Loading scoring pipeline...")
def load_trained_model():
    """Load model and inspect scaler and logistic coefficients."""
    if not MODEL_FILE.exists():
        st.error(f"Model file not found at: {MODEL_FILE.resolve()}")
        st.stop()
    model = joblib.load(MODEL_FILE)
    scaler = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    return model, scaler, lr


pipeline, scaler, lr_model = load_trained_model()

# Extract model math parameters
mean_hours, mean_att = scaler.mean_[0], scaler.mean_[1]
scale_hours, scale_att = scaler.scale_[0], scaler.scale_[1]
coef_hours, coef_att = lr_model.coef_[0][0], lr_model.coef_[0][1]
intercept = lr_model.intercept_[0]

# Unscaled formula: log_odds = b0 + w1*hours + w2*attendance
w1 = coef_hours / scale_hours
w2 = coef_att / scale_att
b0 = intercept - (coef_hours * mean_hours / scale_hours) - (coef_att * mean_att / scale_att)


def predict_probability(hrs: float, att: float) -> float:
    """Return P(Pass) for given hours and attendance."""
    return float(pipeline.predict_proba([[hrs, att]])[0][1])


# Initialize session state defaults if absent
if "hours_input" not in st.session_state:
    st.session_state["hours_input"] = 5.0
if "att_input" not in st.session_state:
    st.session_state["att_input"] = 80
if "name_input" not in st.session_state:
    st.session_state["name_input"] = "Sam Morgan"

# -----------------------------------------------------------------------------
# 3. Sidebar Controls & Presets
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Studio Controls")
    st.caption("Configure global scoring parameters & quick archetypes.")

    st.markdown("---")
    st.markdown("##### ⚡ Quick Student Archetypes")
    st.caption("Instantly load realistic academic profiles:")

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        if st.button("🌟 Honor Roll", width="stretch"):
            st.session_state["hours_input"] = 8.5
            st.session_state["att_input"] = 95
            st.session_state["name_input"] = "Emma Vance"
        if st.button("⚖️ Average", width="stretch"):
            st.session_state["hours_input"] = 4.5
            st.session_state["att_input"] = 75
            st.session_state["name_input"] = "Alex Chen"
    with c_p2:
        if st.button("⚠️ At Risk", width="stretch"):
            st.session_state["hours_input"] = 2.5
            st.session_state["att_input"] = 60
            st.session_state["name_input"] = "Jordan Riley"
        if st.button("🚨 Critical", width="stretch"):
            st.session_state["hours_input"] = 1.0
            st.session_state["att_input"] = 40
            st.session_state["name_input"] = "Sam Miller"

    st.markdown("---")
    st.markdown("##### 🎯 Classification Settings")
    decision_threshold = st.slider(
        "Decision Threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Standard threshold is 0.50. Raise to enforce higher certainty before declaring PASS.",
    )

    st.markdown("---")
    st.markdown("##### 📦 Model Health")
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
            <div>• <b>Pipeline:</b> Scaler + LogisticReg</div>
            <div>• <b>Persistence:</b> Joblib Binary</div>
            <div>• <b>Status:</b> <span style="color: #34D399; font-weight: 600;">Active & Synced</span></div>
            <div>• <b>Decision Boundary:</b> P ≥ 50%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Initialize session state defaults if absent

# -----------------------------------------------------------------------------
# 4. Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">⚡ Machine Learning Academic Predictor</div>
        <h1 class="hero-title">Student Pass / Fail Studio</h1>
        <p class="hero-subtitle">
            Evaluate academic success probability in real-time using our standardized 
            logistic regression engine. Explore sensitivity landscapes, decision thresholds, 
            and targeted academic intervention strategies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 5. Tabbed Application Workflow
# -----------------------------------------------------------------------------
tab_predict, tab_boundary, tab_sensitivity, tab_cohort, tab_math = st.tabs(
    [
        "🎯 Individual Assessment",
        "🗺️ 2D Decision Space",
        "📈 Sensitivity & Curves",
        "👥 Class Cohort Simulator",
        "🧠 Model Architecture",
    ]
)

# =============================================================================
# TAB 1: INDIVIDUAL ASSESSMENT
# =============================================================================
with tab_predict:
    col_input, col_result = st.columns([1.05, 1.35], gap="large")

    with col_input:
        st.markdown("### 📋 Student Profile")
        student_name = st.text_input(
            "Student Name / ID",
            placeholder="e.g. Sam Morgan",
            key="name_input",
        )

        st.markdown("---")
        st.markdown("##### ⏱️ Weekly Study Dedication")
        study_hours = st.slider(
            "Study Hours per Week",
            min_value=0.0,
            max_value=12.0,
            step=0.25,
            key="hours_input",
            help="Weekly independent study and lab preparation time.",
        )

        st.markdown("##### 🏫 Lecture & Lab Attendance")
        attendance = st.slider(
            "Attendance Rate (%)",
            min_value=0,
            max_value=100,
            step=1,
            key="att_input",
            help="Cumulative recorded attendance across all course sessions.",
        )

        display_name = student_name.strip() if student_name.strip() else "Selected Student"

        # Quick stats pill under input
        st.markdown("<br>", unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-pill-label">Daily Study Avg</div>
                    <div class="stat-pill-value">{(study_hours / 7):.1f}h</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_s2:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-pill-label">Absence Estimate</div>
                    <div class="stat-pill-value">{100 - attendance}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_result:
        # Calculate prediction dynamically
        prob_pass = predict_probability(study_hours, attendance)
        is_pass = prob_pass >= decision_threshold
        pct_display = prob_pass * 100

        # Confidence categorization
        if prob_pass >= 0.80:
            confidence_tag = "High Confidence Pass"
            tag_color = "#34D399"
        elif prob_pass >= decision_threshold:
            confidence_tag = "Moderate Pass"
            tag_color = "#6EE7B7"
        elif prob_pass >= 0.35:
            confidence_tag = "Borderline / Moderate Risk"
            tag_color = "#FCD34D"
        else:
            confidence_tag = "High Academic Risk"
            tag_color = "#F87171"

        st.markdown("### 📊 Predictive Outcome")

        # Result Banner Card
        if is_pass:
            banner_html = f"""
            <div class="verdict-box-pass">
                <span style="color: #6EE7B7; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                    Final Assessment Result
                </span>
                <div class="verdict-tag verdict-pass-text">PASS</div>
                <div style="color: #E2E8F0; font-size: 1.05rem; font-weight: 500;">
                    <b>{display_name}</b> is on track to successfully complete the course.
                </div>
                <div class="prob-meter-container">
                    <div class="prob-meter-fill-pass" style="width: {pct_display:.1f}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">
                    <span>Pass Confidence: <b style="color: #34D399;">{pct_display:.1f}%</b></span>
                    <span>Threshold: {decision_threshold*100:.0f}%</span>
                </div>
            </div>
            """
        else:
            banner_html = f"""
            <div class="verdict-box-fail">
                <span style="color: #FCA5A5; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                    Final Assessment Result
                </span>
                <div class="verdict-tag verdict-fail-text">FAIL</div>
                <div style="color: #E2E8F0; font-size: 1.05rem; font-weight: 500;">
                    <b>{display_name}</b> is predicted to be at critical risk of failure.
                </div>
                <div class="prob-meter-container">
                    <div class="prob-meter-fill-fail" style="width: {pct_display:.1f}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">
                    <span>Pass Confidence: <b style="color: #F87171;">{pct_display:.1f}%</b></span>
                    <span>Threshold: {decision_threshold*100:.0f}%</span>
                </div>
            </div>
            """

        st.markdown(banner_html, unsafe_allow_html=True)

        # 3 Detailed Metric Cards
        col_m1, col_m2, col_m3 = st.columns(3)
        odds = (prob_pass / (1.0 - prob_pass)) if prob_pass < 0.999 else 999.0

        with col_m1:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-pill-label">P(Pass)</div>
                    <div class="stat-pill-value" style="color: {tag_color};">{prob_pass:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-pill-label">Risk Profile</div>
                    <div class="stat-pill-value" style="font-size: 0.95rem; color: {tag_color}; padding-top: 5px;">
                        {confidence_tag}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-pill-label">Pass Odds</div>
                    <div class="stat-pill-value">{min(odds, 99.9):.1f}:1</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Actionable AI Recommendation Engine
        target_prob = 0.75
        target_log_odds = np.log(target_prob / (1.0 - target_prob))
        req_hours = (target_log_odds - b0 - w2 * attendance) / w1

        if is_pass:
            if prob_pass >= 0.85:
                rec_text = f"✨ <b>Outstanding Performance:</b> {display_name} demonstrates strong consistency. Maintaining current habits ({study_hours:.1f} hrs/wk, {attendance}% attendance) ensures high likelihood of academic distinction."
            else:
                rec_text = f"💡 <b>Solid Foundation:</b> {display_name} is passing comfortably. An additional <b>0.5 - 1.0 hour</b> of weekly review would solidify knowledge retention and buffer against exam variance."
        else:
            if req_hours > study_hours and req_hours <= 12.0:
                hours_diff = req_hours - study_hours
                rec_text = f"🎯 <b>Prescriptive Action:</b> Increasing study time by <b>+{hours_diff:.1f} hrs/week</b> (target: {req_hours:.1f} hrs) at current {attendance}% attendance would lift pass probability above <b>75%</b>."
            else:
                rec_text = f"⚠️ <b>Multi-Factor Support Needed:</b> Attendance is low ({attendance}%). Recommend scheduling an academic advisory meeting and raising attendance alongside study hours to recover passing standing."

        st.markdown(f'<div class="rec-box">{rec_text}</div>', unsafe_allow_html=True)

        with st.expander("🔍 Model Inference Breakdown", expanded=False):
            st.markdown(
                f"""
                - **Standardized Hours:** `({study_hours} - {mean_hours:.2f}) / {scale_hours:.2f} = {((study_hours - mean_hours) / scale_hours):.3f}`
                - **Standardized Attendance:** `({attendance} - {mean_att:.2f}) / {scale_att:.2f} = {((attendance - mean_att) / scale_att):.3f}`
                - **Calculated Log-Odds:** `{b0:.2f} + ({w1:.3f} × {study_hours}) + ({w2:.4f} × {attendance}) = {(b0 + w1*study_hours + w2*attendance):.3f}`
                - **Sigmoid Output P(Pass):** `1 / (1 + e^(-log_odds)) = {prob_pass:.4f}`
                """
            )

# =============================================================================
# TAB 2: 2D DECISION BOUNDARY
# =============================================================================
with tab_boundary:
    st.markdown("### 🗺️ Course Decision Space & Boundary Map")
    st.write(
        "Explore how study hours and attendance interact across the entire feature domain. "
        "The white line marks the 50% pass threshold. The **glowing purple marker** indicates your active student."
    )

    # Generate synthetic grid for contour visualization
    grid_hours = np.linspace(0, 12, 50)
    grid_att = np.linspace(20, 100, 50)
    H, A = np.meshgrid(grid_hours, grid_att)
    flat_H = H.ravel()
    flat_A = A.ravel()

    # Vectorized probability calculation using unscaled linear log-odds
    grid_log_odds = b0 + w1 * flat_H + w2 * flat_A
    grid_probs = 1.0 / (1.0 + np.exp(-grid_log_odds))

    df_grid = pd.DataFrame({
        "Study Hours": flat_H,
        "Attendance (%)": flat_A,
        "Pass Probability": grid_probs,
    })

    # Decision line: log_odds = 0 => attendance = (-b0 - w1 * hours) / w2
    line_hours = np.linspace(0, 12, 100)
    line_att = (-b0 - w1 * line_hours) / w2
    valid_mask = (line_att >= 0) & (line_att <= 100)
    df_boundary = pd.DataFrame({
        "Study Hours": line_hours[valid_mask],
        "Attendance (%)": line_att[valid_mask],
    })

    # Active student marker
    df_student = pd.DataFrame([{
        "Study Hours": study_hours,
        "Attendance (%)": attendance,
        "Label": f"{display_name} ({prob_pass*100:.1f}%)",
    }])

    # Heatmap background
    heatmap = (
        alt.Chart(df_grid)
        .mark_rect()
        .encode(
            x=alt.X("Study Hours:Q", bin=alt.Bin(maxbins=50), title="Study Hours / Week"),
            y=alt.Y("Attendance (%):Q", bin=alt.Bin(maxbins=50), title="Attendance Percentage (%)"),
            color=alt.Color(
                "Pass Probability:Q",
                scale=alt.Scale(
                    domain=[0, 0.5, 1.0],
                    range=["#EF4444", "#F59E0B", "#10B981"],
                ),
                legend=alt.Legend(title="P(Pass)"),
            ),
            tooltip=[
                alt.Tooltip("Study Hours:Q", format=".1f"),
                alt.Tooltip("Attendance (%):Q", format=".0f"),
                alt.Tooltip("Pass Probability:Q", format=".2%"),
            ],
        )
    )

    # 50% Decision Boundary Line
    boundary_line = (
        alt.Chart(df_boundary)
        .mark_line(color="#FFFFFF", strokeWidth=3, strokeDash=[6, 4])
        .encode(x="Study Hours:Q", y="Attendance (%):Q")
    )

    # Active Student Point
    student_point = (
        alt.Chart(df_student)
        .mark_circle(size=260, color="#6366F1", stroke="#FFFFFF", strokeWidth=2.5)
        .encode(
            x="Study Hours:Q",
            y="Attendance (%):Q",
            tooltip=[
                alt.Tooltip("Label:N"),
                alt.Tooltip("Study Hours:Q"),
                alt.Tooltip("Attendance (%):Q"),
            ],
        )
    )

    chart_boundary = (heatmap + boundary_line + student_point).properties(
        width="container",
        height=420,
    )

    st.altair_chart(chart_boundary, width="stretch")
    st.caption("Dashed white line = 50% Pass Decision Boundary. Green area = High probability of passing. Red area = High risk of failing.")

# =============================================================================
# TAB 3: SENSITIVITY & WHAT-IF CURVES
# =============================================================================
with tab_sensitivity:
    st.markdown("### 📈 Sensitivity & Sweep Curves")
    st.write(
        "Analyze how pass probabilities transition across continuous sweeps of each feature, "
        "holding the other variable constant at your student's current value."
    )

    col_c1, col_c2 = st.columns(2, gap="large")

    with col_c1:
        st.markdown(f"#### Study Hours Sweep *(Attendance fixed at {attendance}%)*")
        hrs_range = np.linspace(0.0, 12.0, 80)
        p_hrs = [predict_probability(h, attendance) for h in hrs_range]
        df_sweep_h = pd.DataFrame({"Study Hours": hrs_range, "Pass Probability": p_hrs})

        area_h = (
            alt.Chart(df_sweep_h)
            .mark_area(
                line={"color": "#6366F1", "width": 3},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(99, 102, 241, 0.4)", offset=0),
                        alt.GradientStop(color="rgba(99, 102, 241, 0.02)", offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0,
                ),
            )
            .encode(
                x=alt.X("Study Hours:Q", title="Weekly Study Hours"),
                y=alt.Y("Pass Probability:Q", title="P(Pass)", scale=alt.Scale(domain=[0, 1])),
                tooltip=["Study Hours:Q", alt.Tooltip("Pass Probability:Q", format=".2%")],
            )
        )

        rule_thresh = (
            alt.Chart(pd.DataFrame({"y": [decision_threshold]}))
            .mark_rule(color="#EF4444", strokeDash=[4, 4])
            .encode(y="y:Q")
        )

        # Highlight current student point
        cur_h_point = (
            alt.Chart(pd.DataFrame({"Study Hours": [study_hours], "Pass Probability": [prob_pass]}))
            .mark_circle(size=140, color="#10B981", stroke="#FFFFFF", strokeWidth=2)
            .encode(x="Study Hours:Q", y="Pass Probability:Q")
        )

        st.altair_chart((area_h + rule_thresh + cur_h_point).properties(height=320), width="stretch")

    with col_c2:
        st.markdown(f"#### Attendance Sweep *(Study Hours fixed at {study_hours:.1f} hrs)*")
        att_range = np.linspace(0.0, 100.0, 80)
        p_att = [predict_probability(study_hours, a) for a in att_range]
        df_sweep_a = pd.DataFrame({"Attendance (%)": att_range, "Pass Probability": p_att})

        area_a = (
            alt.Chart(df_sweep_a)
            .mark_area(
                line={"color": "#10B981", "width": 3},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(16, 185, 129, 0.4)", offset=0),
                        alt.GradientStop(color="rgba(16, 185, 129, 0.02)", offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0,
                ),
            )
            .encode(
                x=alt.X("Attendance (%):Q", title="Attendance Percentage (%)"),
                y=alt.Y("Pass Probability:Q", title="P(Pass)", scale=alt.Scale(domain=[0, 1])),
                tooltip=["Attendance (%):Q", alt.Tooltip("Pass Probability:Q", format=".2%")],
            )
        )

        cur_a_point = (
            alt.Chart(pd.DataFrame({"Attendance (%)": [attendance], "Pass Probability": [prob_pass]}))
            .mark_circle(size=140, color="#6366F1", stroke="#FFFFFF", strokeWidth=2)
            .encode(x="Attendance (%):Q", y="Pass Probability:Q")
        )

        st.altair_chart((area_a + rule_thresh + cur_a_point).properties(height=320), width="stretch")

# =============================================================================
# TAB 4: CLASS COHORT SIMULATOR
# =============================================================================
with tab_cohort:
    st.markdown("### 👥 Classroom Cohort Simulator")
    st.write(
        "Simulate batch scoring across an entire roster of students to analyze overall pass rate "
        "distributions, grade risk flags, and intervention targets."
    )

    col_sim_ctrl1, col_sim_ctrl2, col_sim_ctrl3 = st.columns(3)
    with col_sim_ctrl1:
        cohort_size = st.slider("Cohort Size", min_value=10, max_value=80, value=30, step=5)
    with col_sim_ctrl2:
        class_rigor = st.select_slider(
            "Study Habit Profile",
            options=["Underprepared", "Balanced Mix", "High Achievers"],
            value="Balanced Mix",
        )
    with col_sim_ctrl3:
        random_seed = st.number_input("Random Seed", value=42, step=1)

    # Generate synthetic cohort based on profile
    rng = np.random.default_rng(random_seed)
    if class_rigor == "Underprepared":
        h_vals = np.clip(rng.normal(3.2, 1.8, cohort_size), 0.5, 11.5)
        a_vals = np.clip(rng.normal(62, 18, cohort_size), 20, 100)
    elif class_rigor == "High Achievers":
        h_vals = np.clip(rng.normal(6.5, 2.0, cohort_size), 1.5, 12.0)
        a_vals = np.clip(rng.normal(88, 10, cohort_size), 50, 100)
    else:
        h_vals = np.clip(rng.normal(4.8, 2.4, cohort_size), 0.5, 12.0)
        a_vals = np.clip(rng.normal(76, 16, cohort_size), 30, 100)

    cohort_probs = [predict_probability(h, a) for h, a in zip(h_vals, a_vals)]
    cohort_verdicts = ["PASS" if p >= decision_threshold else "FAIL" for p in cohort_probs]

    student_names = [f"Student {i+1:02d}" for i in range(cohort_size)]
    df_cohort = pd.DataFrame({
        "Student": student_names,
        "Study Hours": np.round(h_vals, 1),
        "Attendance (%)": np.round(a_vals, 0).astype(int),
        "Pass Probability": np.round(cohort_probs, 4),
        "Verdict": cohort_verdicts,
    })

    pass_count = sum(1 for v in cohort_verdicts if v == "PASS")
    pass_pct = (pass_count / cohort_size) * 100

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-pill-label">Projected Pass Rate</div>
                <div class="stat-pill-value" style="color: {'#34D399' if pass_pct >= 60 else '#F87171'};">
                    {pass_pct:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_stat2:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-pill-label">Passing Count</div>
                <div class="stat-pill-value">{pass_count} / {cohort_size}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_stat3:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-pill-label">Avg Study Dedication</div>
                <div class="stat-pill-value">{np.mean(h_vals):.1f}h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_stat4:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-pill-label">Avg Attendance</div>
                <div class="stat-pill-value">{np.mean(a_vals):.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_plot, col_tbl = st.columns([1.1, 1], gap="large")

    with col_plot:
        st.markdown("##### Distribution of Pass Probability")
        hist_chart = (
            alt.Chart(df_cohort)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X(
                    "Pass Probability:Q",
                    bin=alt.Bin(maxbins=15),
                    title="Predicted Pass Probability",
                ),
                y=alt.Y("count()", title="Number of Students"),
                color=alt.Color(
                    "Verdict:N",
                    scale=alt.Scale(domain=["PASS", "FAIL"], range=["#10B981", "#EF4444"]),
                ),
            )
            .properties(height=280)
        )
        st.altair_chart(hist_chart, width="stretch")

    with col_tbl:
        st.markdown("##### Cohort Roster")
        st.dataframe(
            df_cohort.style.format({
                "Pass Probability": "{:.1%}",
                "Study Hours": "{:.1f}",
                "Attendance (%)": "{:d}%",
            }),
            height=280,
            width="stretch",
        )

# =============================================================================
# TAB 5: MODEL ARCHITECTURE & WEIGHTS
# =============================================================================
with tab_math:
    st.markdown("### 🧠 Fitted Model Architecture")
    st.write(
        "Inspect the standardized weights, scaler moments, and mathematical basis "
        "persisted inside `student_pass_model.joblib`."
    )

    col_w1, col_w2 = st.columns(2, gap="large")

    with col_w1:
        st.markdown("##### Standardized Feature Weights (Logistic Regression)")
        weights_df = pd.DataFrame({
            "Feature": ["Weekly Study Hours", "Attendance Percentage"],
            "Standardized Coefficient": [coef_hours, coef_att],
            "Relative Impact": ["Higher Impact", "Moderate Impact"],
        })

        feat_bar = (
            alt.Chart(weights_df)
            .mark_bar(cornerRadius=6, size=28)
            .encode(
                x=alt.X("Standardized Coefficient:Q", title="Weight (Normalized β)"),
                y=alt.Y("Feature:N", sort="-x", title=""),
                color=alt.Color(
                    "Feature:N",
                    scale=alt.Scale(range=["#6366F1", "#10B981"]),
                    legend=None,
                ),
                tooltip=["Feature:N", alt.Tooltip("Standardized Coefficient:Q", format=".3f")],
            )
            .properties(height=160)
        )
        st.altair_chart(feat_bar, width="stretch")

        st.caption(
            f"**Intercept (β0):** `{intercept:.4f}` • "
            f"**Hours Weight:** `{coef_hours:.4f}` • "
            f"**Attendance Weight:** `{coef_att:.4f}`"
        )

    with col_w2:
        st.markdown("##### StandardScaler Baseline Normalization")
        st.markdown(
            f"""
            Before training the linear classifier, the input dimensions were standardized 
            to mean 0 and unit variance:

            | Feature | Training Mean (μ) | Std Dev (σ) |
            | :--- | :--- | :--- |
            | **Study Hours** | `{mean_hours:.2f} hrs` | `{scale_hours:.2f} hrs` |
            | **Attendance** | `{mean_att:.2f}%` | `{scale_att:.2f}%` |

            **Unscaled Closed-Form Log-Odds:**
            $$\\text{{log-odds}} = {b0:.3f} + ({w1:.4f} \\times \\text{{hours}}) + ({w2:.4f} \\times \\text{{attendance}})$$
            """
        )

    st.divider()
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #64748B; text-align: center; padding: 10px 0;">
            EduPredict Studio • Built with Streamlit, Scikit-Learn & Altair • Model persistence via Joblib
        </div>
        """,
        unsafe_allow_html=True,
    )
