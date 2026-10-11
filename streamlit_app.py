import streamlit as st
import joblib
import os
import pandas as pd

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Student Predictor | ML Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "student_pass_model.joblib"
)

model = joblib.load(MODEL_PATH)

# ==========================================
# CUSTOM IT DASHBOARD DESIGN
# ==========================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: #07111f;
        color: #e5edf8;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1728, #07111f);
        border-right: 1px solid #1b3552;
    }

    [data-testid="stSidebar"] h1 {
        color: #ff7900;
        font-size: 25px;
    }

    /* Header */
    .main-title {
        font-size: clamp(28px, 3vw, 42px);
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1.2;
    }

    .orange {
        color: #ff7900;
    }

    .subtitle {
        color: #91b5d8;
        font-size: 16px;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* Section headings */
    h2, h3 {
        color: #e5edf8 !important;
    }

    /* Cards */
    .custom-card {
        background: linear-gradient(135deg, #0d2035, #102943);
        border: 1px solid #244665;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 15px;
    }

    .section-heading {
        color: #91cfff;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    /* Prediction cards */
    .pass-card {
        background: linear-gradient(135deg, #063b32, #082c30);
        border: 1px solid #00d6a0;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }

    .fail-card {
        background: linear-gradient(135deg, #421e29, #281b2c);
        border: 1px solid #ff5a65;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }

    .result-label {
        color: #a8bfd3;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .pass-result {
        color: #00e6a8;
        font-size: 38px;
        font-weight: 800;
        margin-top: 8px;
    }

    .fail-result {
        color: #ff6872;
        font-size: 38px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* Information panel */
    .info-card {
        background: #10243a;
        border: 1px solid #244665;
        border-left: 4px solid #ff7900;
        border-radius: 10px;
        padding: 18px;
        color: #b8d3ed;
        line-height: 1.7;
    }

    /* Input fields */
    .stTextInput label,
    .stNumberInput label {
        color: #b8d3ed !important;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #ff7900, #ff5100);
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        min-height: 45px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #ff9638;
        color: white;
        border: none;
        transform: translateY(-1px);
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #0d2035;
        border: 1px solid #244665;
        border-radius: 12px;
        padding: 17px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9fc8eb;
    }

    /* Progress bars */
    div[data-testid="stProgress"] > div {
        background-color: #1b3552;
        border-radius: 10px;
    }

    div[data-testid="stProgress"] > div > div {
        background-color: #ff7900;
        border-radius: 10px;
    }

    /* Dividers */
    hr {
        border-color: #1b3552;
    }

    /* Sidebar text */
    .sidebar-caption {
        color: #7597b9;
        font-size: 12px;
        letter-spacing: 1px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6483a3;
        font-size: 12px;
        padding: 20px 0;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("# 🎓 Student")
    st.markdown("## :orange[Pass/Fail Predictor]")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "📊 Dashboard",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.markdown(
        '<p class="sidebar-caption">MACHINE LEARNING SYSTEM</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="color:#91b5d8; margin-top:35px; line-height:1.8;">
        "Small steps in study lead to big results in the future."
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Version 1.0.0")

# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="main-title">🎓 Student '
    '<span class="orange">Pass/Fail</span> Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'MACHINE LEARNING DASHBOARD &nbsp;|&nbsp; '
    'Better Data. Smarter Decisions.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================
# HOME PAGE
# ==========================================
if page == "🏠 Home":

    input_col, result_col = st.columns(
        [0.9, 1.5],
        gap="large"
    )

    # STUDENT INPUT
    with input_col:
        st.markdown(
            '<div class="section-heading">🧑‍🎓 Student Input</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):

            student_name = st.text_input(
                "Student Name",
                placeholder="Enter student's name"
            )

            study_hours = st.number_input(
                "Study Hours Per Day",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.5
            )

            st.caption(
                "Minimum: 0 hours  |  Maximum: 24 hours"
            )

            predict_clicked = st.button(
                "⚡ Predict Student Result",
                use_container_width=True
            )

    # PREDICTION RESULT
    with result_col:
        st.markdown(
            '<div class="section-heading">🎯 Prediction Result</div>',
            unsafe_allow_html=True
        )

        if predict_clicked:

            if not student_name.strip():
                st.warning("Please enter the student's name.")

            else:
                input_data = [[float(study_hours)]]

                prediction_value = int(
                    model.predict(input_data)[0]
                )

                result = (
                    "Pass"
                    if prediction_value == 1
                    else "Fail"
                )

                # Illustrative scores only
                pass_percentage = round(
                    min(95, max(5, (study_hours / 8) * 100)),
                    2
                )

                fail_percentage = round(
                    100 - pass_percentage,
                    2
                )

                # Prediction card
                if result == "Pass":
                    st.markdown(f"""
                    <div class="pass-card">
                        <div class="result-label">
                            MODEL PREDICTION
                        </div>
                        <div class="pass-result">
                            ✓ {result}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="fail-card">
                        <div class="result-label">
                            MODEL PREDICTION
                        </div>
                        <div class="fail-result">
                            ✕ {result}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")

                info1, info2 = st.columns(2)

                with info1:
                    st.markdown("**👤 Student**")
                    st.write(student_name.strip())

                with info2:
                    st.markdown("**⏱️ Study Hours**")
                    st.write(f"{study_hours:.1f} hours/day")

                st.markdown("### 📈 Study-Based Scores")

                score1, score2 = st.columns(2)

                with score1:
                    st.metric(
                        "🟢 Pass Score",
                        f"{pass_percentage:.2f}%"
                    )
                    st.progress(pass_percentage / 100)

                with score2:
                    st.metric(
                        "🔴 Fail Score",
                        f"{fail_percentage:.2f}%"
                    )
                    st.progress(fail_percentage / 100)

                st.caption(
                    "These percentages are illustrative scores based "
                    "on study hours, not probabilities learned by "
                    "the machine-learning model."
                )

        else:
            st.markdown("""
            <div class="custom-card">
                <h3>⚡ System Ready</h3>
                <p style="color:#91b5d8;">
                    Enter the student's information on the left,
                    then click the prediction button to see the result.
                </p>
                <p style="color:#ff7900;">
                    ● WAITING FOR INPUT
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ======================================
    # STUDY HOURS CHART
    # ======================================
    st.markdown("---")

    st.markdown(
        '<div class="section-heading">'
        '📊 Study Hours and Predicted Class'
        '</div>',
        unsafe_allow_html=True
    )

    hours_list = list(range(1, 9))

    chart_data = pd.DataFrame({
        "Study Hours": hours_list,
        "Predicted Class": [
            int(model.predict([[hours]])[0])
            for hours in hours_list
        ]
    })

    st.bar_chart(
        chart_data.set_index("Study Hours"),
        y="Predicted Class",
        color="#ff7900"
    )

    st.caption(
        "0 = Fail | 1 = Pass. These classifications come from the trained model."
    )

    # ======================================
    # HOW IT WORKS
    # ======================================
    st.markdown(
        '<div class="section-heading">💡 How It Works</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        The Decision Tree model predicts Pass or Fail using study hours.
        The Pass Score and Fail Score are separate illustrative values
        that change with study hours. They are not probabilities learned
        by the model and do not guarantee academic outcomes.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# DASHBOARD PAGE
# ==========================================
elif page == "📊 Dashboard":

    st.markdown("## :orange[Prediction Dashboard]")

    st.write(
        "Explore how the trained model classifies students "
        "based on different study-hour inputs."
    )

    hours_list = list(range(0, 13))

    dashboard_data = pd.DataFrame({
        "Study Hours": hours_list,
        "Predicted Class": [
            int(model.predict([[hours]])[0])
            for hours in hours_list
        ]
    })

    st.markdown("### Model Classification Chart")

    st.line_chart(
        dashboard_data.set_index("Study Hours"),
        color="#ff7900"
    )

    st.markdown("### Prediction Data")

    st.dataframe(
        dashboard_data,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# ABOUT PAGE
# ==========================================
elif page == "ℹ️ About":

    st.markdown("## :orange[About This Project]")

    st.markdown("""
    <div class="custom-card">
        <h3>Student Pass/Fail Prediction System</h3>
        <p>
            This application uses a trained Decision Tree machine-learning
            model to predict whether a student will Pass or Fail based
            on daily study hours.
        </p>
        <br>
        <b>Technology Stack</b>
        <p>Python · Streamlit · Pandas · Scikit-learn · Joblib</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "The displayed percentages are illustrative scores. "
        "They are not actual probabilities generated by the model."
    )

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")

st.markdown("""
<div class="footer">
    STUDENT PASS/FAIL PREDICTOR &nbsp;|&nbsp;
    MACHINE LEARNING PROJECT
</div>
""", unsafe_allow_html=True)