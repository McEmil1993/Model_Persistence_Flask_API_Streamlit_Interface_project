
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="StudentPass | Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f4f7fc;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10264d 0%, #173b70 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    color: white;
    padding: 12px 0 5px 0;
}

.sidebar-caption {
    color: #c5d7f3;
    font-size: 12px;
    margin-bottom: 28px;
}

.hero {
    background: linear-gradient(120deg, #173b70, #2878df);
    padding: 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 10px 28px rgba(35, 93, 170, 0.15);
}

.hero h1 {
    color: white;
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #e0edff;
    font-size: 15px;
    margin-top: 10px;
    margin-bottom: 0;
}

.section-title {
    color: #172b4d;
    font-size: 21px;
    font-weight: 750;
    margin: 8px 0 16px 0;
}

.panel {
    background: white;
    border: 1px solid #e5eaf3;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 5px 18px rgba(31, 54, 88, 0.04);
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    border: 1px solid #e5eaf3;
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #6b7a90;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #173b70;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

div.stButton > button {
    background: linear-gradient(90deg, #2165c5, #3284ed);
    color: white;
    border: none;
    border-radius: 11px;
    padding: 13px 18px;
    font-weight: 700;
    min-height: 48px;
    width: 100%;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: #174e9f;
    color: white;
    border: none;
}

div[data-testid="stForm"] {
    background: white;
    padding: 22px;
    border: 1px solid #e5eaf3;
    border-radius: 18px;
}

.footer {
    text-align: center;
    color: #8491a5;
    font-size: 12px;
    padding: 25px 0 10px 0;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="brand">🎓 StudentPass</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-caption">STUDENT PERFORMANCE SYSTEM</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 🏠 Dashboard")
    st.caption("Student prediction and performance overview")

    st.markdown("---")
    st.markdown("### 📊 Prediction Tool")
    st.caption("Estimate a result from study hours")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "A machine-learning demonstration using a saved "
        "Scikit-learn model."
    )

    st.markdown("---")
    st.caption("Version 1.0 • StudentPass")

# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>Student Performance Dashboard</h1>
    <p>
        Welcome! Enter student information below to estimate
        a possible academic outcome using the trained model.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- TOP METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🤖 MODEL STATUS</div>
        <div class="metric-value">Ready</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">📚 INPUT FEATURE</div>
        <div class="metric-value">Study Hours</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🎯 OUTPUT CLASSES</div>
        <div class="metric-value">Pass / Fail</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- MAIN CONTENT ----------------
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown(
        '<div class="section-title">📝 Student Information</div>',
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):
        student_name = st.text_input(
            "Student Name",
            placeholder="Enter student's full name"
        )

        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
            help="Enter a value from 0 to 24 hours."
        )

        st.caption("💡 Tip: Use the student's estimated daily study hours.")

        submitted = st.form_submit_button(
            "🔍 Predict Student Result",
            use_container_width=True
        )

    if submitted:
        if not student_name.strip():
            st.warning("Please enter the student's name.")
        else:
            prediction = model.predict([[study_hours]])[0]

            st.session_state["prediction_result"] = {
                "name": student_name.strip(),
                "hours": study_hours,
                "prediction": str(prediction)
            }

with right:
    st.markdown(
        '<div class="section-title">📈 Prediction Overview</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.get("prediction_result")

    if result:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.caption("PREDICTION FOR")
        st.subheader(result["name"])

        if result["prediction"] == "Pass":
            st.success("🎉 Predicted Result: PASS")
            st.markdown(
                "The model classifies this study-hours input as **Pass**."
            )
        else:
            st.error("📘 Predicted Result: FAIL")
            st.markdown(
                "The model classifies this study-hours input as **Fail**."
            )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Study Hours", f'{result["hours"]:.1f} hrs')
        with m2:
            st.metric("Predicted Class", result["prediction"])

        st.progress(min(result["hours"] / 24, 1.0))
        st.caption("Study hours relative to a 24-hour day.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Chart
        st.markdown("#### Study Hours Visualization")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(
            ["Daily Study Hours"],
            [result["hours"]],
            color="#2878df",
            width=0.48
        )
        ax.set_ylim(0, 24)
        ax.set_ylabel("Hours")
        ax.set_title("Student Study Time")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.markdown("""
        <div class="panel">
            <h3 style="color:#173b70;">Ready to predict?</h3>
            <p style="color:#6b7a90;">
                Submit the student information using the form.
                The predicted class and study-hours chart will
                appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.info(
            "Your first prediction will appear here after "
            "you submit the form."
        )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    StudentPass Prediction System · Machine Learning Project<br>
    For educational demonstration only — predictions are not guarantees.
</div>
""", unsafe_allow_html=True)
