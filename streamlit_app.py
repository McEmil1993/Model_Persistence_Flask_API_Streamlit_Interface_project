
import streamlit as st
import joblib
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path(__file__).parent / "student_pass_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

# ---------------- CUSTOM GRADIENT DESIGN ----------------
st.markdown("""
<style>
/* Gradient background */
.stApp {
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 50%,
        #f093fb 100%
    );
    background-attachment: fixed;
}

/* Hide sidebar */
section[data-testid="stSidebar"] {
    display: none;
}

[data-testid="collapsedControl"] {
    display: none;
}

/* Main content width */
.block-container {
    max-width: 800px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */
.main-title {
    color: white;
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
    text-shadow: 0 3px 12px rgba(0,0,0,0.20);
}

.subtitle {
    color: #f5f3ff;
    text-align: center;
    font-size: 16px;
    margin-bottom: 28px;
}

/* Form and result containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 12px 35px rgba(31, 20, 80, 0.18);
}

/* Headings and labels */
h2, h3, label, .stMarkdown p {
    color: #28234a;
}

/* Input fields */
.stTextInput input {
    border-radius: 10px;
    border: 1px solid #c4b5fd;
    padding: 12px;
}

/* Slider color */
.stSlider [data-baseweb="slider"] div {
    color: #6d28d9;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    color: white;
    font-weight: 700;
    background: linear-gradient(90deg, #6d28d9, #9333ea, #ec4899);
    transition: 0.2s ease;
}

.stButton > button:hover {
    color: white;
    border: none;
    box-shadow: 0 6px 18px rgba(109,40,217,0.35);
    transform: translateY(-2px);
}

/* Result text */
.result-pass {
    color: #047857;
    font-size: 30px;
    font-weight: 800;
    text-align: center;
}

.result-fail {
    color: #b91c1c;
    font-size: 30px;
    font-weight: 800;
    text-align: center;
}

/* Footer */
.footer {
    color: white;
    text-align: center;
    font-size: 12px;
    margin-top: 25px;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🎓 Student Pass Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover your predicted academic outcome using machine learning.</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT FORM ----------------
with st.container(border=True):
    st.subheader("📝 Student Information")
    st.write("Enter your details below to generate a prediction.")

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter your full name"
    )

    study_hours = st.slider(
        "Daily Study Hours",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
        help="Select how many hours you study per day."
    )

    st.markdown("---")
    st.write(f"📚 **Study time selected:** {study_hours:g} hours per day")

    predict_clicked = st.button("✨ Predict My Result")

# ---------------- PREDICTION ----------------
if predict_clicked:
    if not student_name.strip():
        st.warning("Please enter the student's name first.")
    else:
        try:
            prediction = model.predict([[study_hours]])[0]

            # Find the probability associated with class 1 (Pass)
            pass_probability = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([[study_hours]])[0]
                classes = list(model.classes_)

                if 1 in classes:
                    pass_probability = float(probabilities[classes.index(1)])
                elif "Pass" in classes:
                    pass_probability = float(
                        probabilities[classes.index("Pass")]
                    )

            is_pass = prediction == 1 or str(prediction).lower() == "pass"

            st.markdown("---")

            with st.container(border=True):
                st.subheader("📊 Prediction Result")
                st.write(f"**Student:** {student_name.strip()}")
                st.write(f"**Daily study hours:** {study_hours:g}")

                if is_pass:
                    st.markdown(
                        '<div class="result-pass">🎉 Predicted: PASS</div>',
                        unsafe_allow_html=True
                    )
                    st.success("The model predicts a passing result.")
                else:
                    st.markdown(
                        '<div class="result-fail">📘 Predicted: FAIL</div>',
                        unsafe_allow_html=True
                    )
                    st.warning("The model predicts a failing result.")

                if pass_probability is not None:
                    st.metric(
                        "Estimated Probability of Passing",
                        f"{pass_probability:.1%}"
                    )
                    st.progress(pass_probability)

                st.caption(
                    "This is a machine-learning prediction, not a guarantee "
                    "of actual academic performance."
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">Powered by Python, Streamlit & Logistic Regression</div>',
    unsafe_allow_html=True
)
