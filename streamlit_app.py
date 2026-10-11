
import streamlit as st
import joblib
import pandas as pd
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FC;
    }

    [data-testid="stSidebar"] {
        background-color: #172554;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #64748B;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .info-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }

    .result-card {
        background: white;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }

    .result-label {
        font-size: 15px;
        color: #64748B;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 36px;
        font-weight: 800;
    }

    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 20px;
        font-weight: 700;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }

    hr {
        border-color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "student_pass_model.joblib"
    )
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the prediction model.")
    st.code(str(e))
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🎓 StudentAI")
    st.caption("STUDENT PERFORMANCE SYSTEM")
    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "Choose a page",
        ["Dashboard", "About"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Prediction Model**")
    st.caption("Logistic Regression")
    st.caption("Student pass prediction")

# ---------------- ABOUT PAGE ----------------
if page == "About":
    st.markdown(
        '<div class="main-title">About StudentAI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Understanding student performance predictions</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 🎯 Purpose
    StudentAI predicts whether a student is likely to pass based on
    their study hours using a trained machine learning model.

    ### ⚙️ How it works
    1. Enter the student's name.
    2. Set the number of study hours.
    3. Click **Predict Result**.
    4. Review the model's prediction.

    ### ⚠️ Important
    Predictions are estimates from a machine learning model. They do
    not guarantee a student's actual academic result.
    """)
    st.stop()

# ---------------- DASHBOARD ----------------
st.markdown(
    '<div class="main-title">Student Pass Prediction</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Analyze study hours and estimate student performance.</div>',
    unsafe_allow_html=True
)

# Summary cards
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <div style="color:#64748B;">📘 SYSTEM</div>
        <h3 style="color:#172554;">StudentAI</h3>
        <div style="color:#64748B;">Prediction dashboard</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <div style="color:#64748B;">🤖 MODEL</div>
        <h3 style="color:#172554;">Machine Learning</h3>
        <div style="color:#64748B;">Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <div style="color:#64748B;">📊 INPUT</div>
        <h3 style="color:#172554;">Study Hours</h3>
        <div style="color:#64748B;">Student information</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown("### 📝 Student Information")

    with st.container(border=True):
        student_name = st.text_input(
            "Student name",
            placeholder="Enter student name"
        )

        study_hours = st.slider(
            "Daily study hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
            help="Adjust the number of hours studied per day."
        )

        st.info(f"Selected study time: **{study_hours:g} hours/day**")

        predict_button = st.button(
            "🔍 Predict Result",
            use_container_width=True
        )

with right:
    st.markdown("### 📋 Prediction Result")

    if predict_button:
        if not student_name.strip():
            st.warning("Please enter the student's name first.")
        else:
            try:
                # This assumes the model was trained using one feature:
                # study hours.
                prediction = model.predict([[study_hours]])[0]

                # Supports common binary labels: 1/0, True/False,
                # or text labels such as Pass/Fail.
                label = str(prediction).strip().lower()

                if label in ("1", "true", "pass", "passed"):
                    result = "PASS"
                    result_color = "#15803D"
                    result_icon = "🎉"
                elif label in ("0", "false", "fail", "failed"):
                    result = "FAIL"
                    result_color = "#B91C1C"
                    result_icon = "📚"
                else:
                    result = str(prediction).upper()
                    result_color = "#1D4ED8"
                    result_icon = "📊"

                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:42px;">{result_icon}</div>
                    <div class="result-label">PREDICTION FOR</div>
                    <h3 style="color:#172554;">{student_name.strip()}</h3>
                    <div class="result-value" style="color:{result_color};">
                        {result}
                    </div>
                    <p style="color:#64748B;">
                        Study hours: {study_hours:g} per day
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Show confidence only when the model supports it.
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([[study_hours]])[0]
                    confidence = max(probabilities) * 100
                    st.metric("Model confidence for its predicted class",
                              f"{confidence:.1f}%")
                    st.caption(
                        "Confidence is not a guarantee of actual student performance."
                    )

                st.session_state["last_prediction"] = {
                    "name": student_name.strip(),
                    "hours": study_hours,
                    "result": result
                }

            except Exception as e:
                st.error("Prediction failed. Check the model's expected input.")
                st.code(str(e))
    else:
        st.markdown("""
        <div class="result-card">
            <div style="font-size:48px;">🎓</div>
            <h3 style="color:#172554;">Ready to Predict</h3>
            <p style="color:#64748B;">
                Enter student information and click Predict Result.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- STUDY HOURS CHART ----------------
st.divider()
st.markdown("### 📈 Study Hours Overview")
st.caption("Explore how different study-hour inputs are classified by your trained model.")

if st.button("Generate Study Hours Chart"):
    try:
        hours = list(range(0, 25))
        predictions = model.predict([[h] for h in hours])

        chart_df = pd.DataFrame({
            "Study Hours": hours,
            "Predicted Class": [str(p) for p in predictions]
        })

        chart_df["Predicted Class"] = chart_df["Predicted Class"].replace({
            "1": "Pass",
            "0": "Fail",
            "True": "Pass",
            "False": "Fail"
        })

        chart_df = chart_df.set_index("Study Hours")
        st.line_chart(chart_df["Predicted Class"].map({
            "Pass": 1,
            "Fail": 0
        }).fillna(0))

        st.caption("Chart values: Pass = 1, Fail = 0. This visualizes model classifications, not guaranteed outcomes.")

    except Exception as e:
        st.error("Unable to generate the chart.")
        st.code(str(e))

st.divider()
st.caption("StudentAI • Machine learning prediction tool • For educational use")
