
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================
# DARK GLASSMORPHISM SAAS THEME
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #070b18;
    --panel: rgba(17, 29, 55, 0.66);
    --border: rgba(112, 160, 255, 0.28);
    --text: #edf4ff;
    --muted: #9aafd3;
    --blue: #399bff;
    --purple: #7954ff;
    --teal: #35f0ce;
    --red: #ff648b;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    color: var(--text);
    background:
        radial-gradient(ellipse at 15% 5%, rgba(27, 76, 151, 0.38), transparent 38%),
        radial-gradient(ellipse at 85% 25%, rgba(91, 43, 166, 0.25), transparent 38%),
        radial-gradient(ellipse at 50% 100%, rgba(0, 125, 150, 0.12), transparent 45%),
        var(--bg);
}

[data-testid="stHeader"] {
    background: rgba(7, 11, 24, 0.3);
}

/* Completely remove sidebar navigation */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* Full-width dashboard */
.block-container {
    max-width: 1600px;
    padding: 2rem 3rem 2rem;
}

h1, h2, h3, p, label {
    color: var(--text);
}

.brand {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.7px;
    color: #f1f6ff;
}

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        115deg,
        rgba(24, 75, 145, 0.45),
        rgba(35, 39, 90, 0.46),
        rgba(12, 110, 126, 0.20)
    );
    border: 1px solid rgba(92, 165, 255, 0.48);
    border-radius: 22px;
    padding: 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06),
                0 15px 45px rgba(0,0,0,0.14);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
}

.eyebrow {
    color: #6ee7ff;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}

.hero h1 {
    font-size: clamp(2rem, 3.5vw, 3.15rem);
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.15;
    margin: 0;
    color: #f2f7ff;
}

.gradient-text {
    background: linear-gradient(90deg, #6edbff, #7f85ff, #c0a1ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #b5c8e9;
    font-size: 0.98rem;
    line-height: 1.7;
    margin-top: 0.8rem;
    margin-bottom: 0;
}

.section-title {
    font-size: 1.13rem;
    font-weight: 750;
    letter-spacing: -0.35px;
    margin-bottom: 0.3rem;
}

.section-subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 1.2rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 27, 51, 0.62);
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.035),
                0 12px 35px rgba(0,0,0,0.12);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

.stTextInput label,
.stNumberInput label {
    color: #d9e7ff !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}

.stTextInput input,
.stNumberInput input {
    background: rgba(8, 17, 37, 0.8) !important;
    color: #f3f7ff !important;
    border: 1px solid rgba(105, 155, 240, 0.38) !important;
    border-radius: 11px !important;
    min-height: 46px;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #529bff !important;
    box-shadow: 0 0 0 3px rgba(57, 155, 255, 0.15) !important;
}

.stTextInput input::placeholder {
    color: #7188ad !important;
}

.stButton > button,
.stFormSubmitButton > button {
    color: white !important;
    background: linear-gradient(105deg, #168fff, #5359ff 60%, #884cff);
    border: 1px solid rgba(154, 186, 255, 0.45);
    border-radius: 11px;
    min-height: 49px;
    font-weight: 700;
    box-shadow: 0 6px 24px rgba(54, 103, 255, 0.24);
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border-color: #7bc8ff;
    box-shadow: 0 0 26px rgba(64, 133, 255, 0.4);
    transform: translateY(-1px);
}

[data-testid="stMetric"] {
    background: rgba(18, 34, 64, 0.75);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 1rem;
}

[data-testid="stMetricLabel"] {
    color: #9eb6dc !important;
}

[data-testid="stMetricValue"] {
    color: #edf5ff !important;
    font-weight: 750;
}

.result-card {
    padding: 1.45rem;
    border-radius: 15px;
    margin: 0.4rem 0 1rem;
    backdrop-filter: blur(16px);
}

.result-pass {
    background: linear-gradient(
        120deg,
        rgba(0, 184, 146, 0.18),
        rgba(18, 53, 71, 0.60)
    );
    border: 1px solid rgba(53, 240, 206, 0.45);
}

.result-fail {
    background: linear-gradient(
        120deg,
        rgba(195, 47, 91, 0.18),
        rgba(58, 28, 57, 0.60)
    );
    border: 1px solid rgba(255, 100, 139, 0.45);
}

.result-label {
    font-size: 0.77rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #b1c5e8;
}

.result-value {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0.35rem 0;
}

.result-pass .result-value {
    color: var(--teal);
    text-shadow: 0 0 22px rgba(53, 240, 206, 0.15);
}

.result-fail .result-value {
    color: var(--red);
    text-shadow: 0 0 22px rgba(255, 100, 139, 0.15);
}

.result-description {
    color: #d0def5;
    font-size: 0.88rem;
    line-height: 1.7;
}

.note {
    background: rgba(34, 91, 166, 0.13);
    border: 1px solid rgba(82, 155, 255, 0.3);
    border-radius: 12px;
    padding: 1rem;
    color: #b9d2f7;
    font-size: 0.82rem;
    line-height: 1.8;
}

.note strong {
    color: #72caff;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}

[data-testid="stAlert"] {
    background: rgba(30, 48, 80, 0.7);
    border-radius: 11px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}

hr {
    border-color: rgba(112, 160, 255, 0.18);
}

.footer {
    text-align: center;
    color: #7f95ba;
    font-size: 0.76rem;
    padding: 1.5rem 0 0.5rem;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .hero {
        padding: 1.4rem;
    }

    .hero h1 {
        font-size: 2rem;
    }

    .result-value {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD SAVED MODEL
# ==================================================
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run: python train_model.py")
    st.stop()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

# ==================================================
# HERO HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ &nbsp; MACHINE LEARNING DASHBOARD</div>
    <h1>
        Student Pass<br>
        <span class="gradient-text">Prediction System</span>
    </h1>
    <p>Predict a student's result based on study hours.</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# MAIN DASHBOARD
# ==================================================
left_col, right_col = st.columns([1.15, 0.95], gap="large")

# ==================================================
# LEFT: FORM AND ANALYTICS
# ==================================================
with left_col:

    st.markdown(
        '<div class="section-title">✦ &nbsp; Enter Student Details</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Fill in the information below to get a prediction.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        with st.form("prediction_form"):
            student_name = st.text_input(
                "Student name",
                placeholder="Enter the student's name",
            )

            study_hours = st.number_input(
                "Study hours",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5,
                help="Enter a value between 0 and 24 hours.",
            )

            st.caption("Enter daily study hours between 0 and 24.")

            submitted = st.form_submit_button(
                "⚡  Predict Result  →",
                use_container_width=True,
            )

        if submitted:
            if not student_name.strip():
                st.warning("Please enter the student's name.")
            else:
                try:
                    prediction = int(model.predict([[study_hours]])[0])

                    st.session_state["last_prediction"] = {
                        "name": student_name.strip(),
                        "hours": study_hours,
                        "prediction": prediction,
                    }
                except Exception as error:
                    st.error(f"Prediction failed: {error}")

    st.markdown("")
    st.markdown(
        '<div class="section-title">▥ &nbsp; Study Hours Analytics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Explore how the model classifies different study-hour values.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        hours_range = list(range(0, 11))
        predicted_classes = [
            int(model.predict([[hours]])[0])
            for hours in hours_range
        ]

        chart_data = pd.DataFrame({
            "Study Hours": hours_range,
            "Predicted Class": predicted_classes,
        })

        fig, ax = plt.subplots(figsize=(8, 4))

        fig.patch.set_facecolor("#101b32")
        ax.set_facecolor("#101b32")

        ax.plot(
            hours_range,
            predicted_classes,
            color="#647aff",
            linewidth=1.8,
            alpha=0.7,
        )

        ax.scatter(
            hours_range,
            predicted_classes,
            c=[
                "#ff648b" if value == 0 else "#35f0ce"
                for value in predicted_classes
            ],
            s=45,
            zorder=3,
        )

        ax.set_xlabel("Daily study hours", color="#a9bedf", labelpad=10)
        ax.set_ylabel("Predicted result", color="#a9bedf")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["FAIL (0)", "PASS (1)"])
        ax.set_xticks(range(0, 11))
        ax.set_ylim(-0.18, 1.18)
        ax.tick_params(colors="#a9bedf", labelsize=9)

        ax.grid(True, color="#324364", alpha=0.4, linewidth=0.7)

        for spine in ax.spines.values():
            spine.set_color("#344969")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Predictions are model classifications, not probabilities "
            "or actual student pass rates."
        )

# ==================================================
# RIGHT: PREDICTION RESULT
# ==================================================
with right_col:

    st.markdown(
        '<div class="section-title">◈ &nbsp; Prediction Result</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Your latest prediction appears here.'
        '</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.get("last_prediction")

    with st.container(border=True):
        if result is None:
            st.markdown("""
            <div style="text-align:center; padding:2.5rem 0;">
                <div style="font-size:3rem;">🔮</div>
                <h3 style="color:#e8f1ff;">Ready to predict?</h3>
                <p style="color:#8fa8d1; line-height:1.8;">
                    Enter student details and click
                    <b>Predict Result</b> to see the outcome.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            name = result["name"]
            hours = result["hours"]
            prediction = result["prediction"]

            is_pass = prediction == 1
            outcome = "PASS" if is_pass else "FAIL"
            css_class = "result-pass" if is_pass else "result-fail"
            symbol = "✓" if is_pass else "✕"

            message = (
                f"Congratulations, {name}! The model predicts a pass."
                if is_pass
                else f"{name}, the model predicts a fail."
            )

            st.markdown(
                f"""
                <div class="result-card {css_class}">
                    <div class="result-label">PREDICTION STATUS</div>
                    <div style="
                        display:flex;
                        align-items:center;
                        gap:14px;
                        margin-top:8px;
                    ">
                        <div style="
                            width:48px;
                            height:48px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            border-radius:50%;
                            font-size:1.8rem;
                            font-weight:800;
                            color:inherit;
                            background:rgba(255,255,255,0.06);
                            border:1px solid currentColor;
                        ">{symbol}</div>
                        <div class="result-value">{outcome}</div>
                    </div>
                    <div class="result-description">{message}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric("Student", name)

            with metric2:
                st.metric("Study hours", f"{hours:g} hrs")

            st.markdown("""
            <div class="note">
                <strong>ⓘ &nbsp; Important Note</strong><br>
                This is a prediction from a simple practice model,
                not a guarantee of the student's actual result.
                The model uses study hours only.
            </div>
            """, unsafe_allow_html=True)

    # ==================================================
    # QUICK INSIGHTS
    # ==================================================
    st.markdown("")
    st.markdown(
        '<div class="section-title">✧ &nbsp; Quick Insights</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Explore the model across 0–10 study hours.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        hours_tested = list(range(0, 11))
        outcomes = [
            int(model.predict([[hours]])[0])
            for hours in hours_tested
        ]

        pass_count = sum(outcomes)
        fail_count = len(outcomes) - pass_count

        q1, q2 = st.columns(2)

        with q1:
            st.metric("Hours tested", len(hours_tested))
            st.metric("PASS classifications", pass_count)

        with q2:
            st.metric("FAIL classifications", fail_count)
            st.metric(
                "Passing classifications",
                f"{pass_count / len(outcomes) * 100:.1f}%",
            )

        st.caption(
            "Percentages describe tested study-hour values, "
            "not the probability that a student will pass."
        )

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    STUDENT PASS PREDICTOR &nbsp; · &nbsp;
    MACHINE LEARNING PRACTICE PROJECT<br><br>
    Built for learning and experimentation
</div>
""", unsafe_allow_html=True)
