import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Student Pass Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            text-align: center;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: #888;
            font-size: 1rem;
            margin-top: 0;
            margin-bottom: 2rem;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .result-pass {
            background: linear-gradient(135deg, #064e3b, #065f46);
            border-left: 6px solid #10b981;
        }
        .result-fail {
            background: linear-gradient(135deg, #7f1d1d, #991b1b);
            border-left: 6px solid #ef4444;
        }
        .result-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            opacity: 0.7;
            margin-bottom: 0.3rem;
        }
        .result-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }
        .metric-box {
            background: rgba(255,255,255,0.04);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .metric-label {
            font-size: 0.8rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 0.3rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<h1 class="main-title">Student Pass Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict whether a student will pass based on study hours</p>', unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ---------------- Constants ----------------
MIN_HOURS = 1.0
MAX_HOURS = 8.0

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a trained Machine Learning model to predict "
        "whether a student will pass or fail based on the number of study hours."
    )
    st.divider()
    st.subheader("Model Info")
    st.write("- Algorithm: Logistic Regression")
    st.write(f"- Trained study-hour range: {MIN_HOURS} - {MAX_HOURS} hrs")
    st.write("- Output: Pass / Fail")

# ---------------- Session State ----------------
if "last_submission" not in st.session_state:
    st.session_state.last_submission = None

# ---------------- Input Form ----------------
st.subheader("Student Information")

col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Student Name", value="", placeholder="e.g., Juan Dela Cruz")
with col2:
    # Allow a large upper bound so a user CAN type something like 150
    # and get a clear rejection message instead of a silent clamp.
    hours = st.number_input(
        "Study Hours (per week)",
        min_value=0.0,
        max_value=1000.0,
        value=5.0,
        step=0.5,
        help=f"The model was trained on {MIN_HOURS} - {MAX_HOURS} hours only.",
    )

# --- Live warning: show as soon as value is out of range (before clicking) ---
hours_out_of_range = hours < MIN_HOURS or hours > MAX_HOURS
if hours_out_of_range:
    st.warning(
        f"Study hours must be between {MIN_HOURS} and {MAX_HOURS}. "
        f"The model was only trained on this range."
    )
    # Also clear any previous result so the old PASS card disappears.
    st.session_state.last_submission = None

st.write("")
predict_clicked = st.button("Predict Result", use_container_width=True, type="primary")

# ---------------- Handle Prediction Click ----------------
if predict_clicked:
    if not name.strip():
        st.error("Please enter the student's name first.")
        st.session_state.last_submission = None
    elif hours_out_of_range:
        st.error(
            f"Cannot predict: {hours} hours is outside the trained range "
            f"({MIN_HOURS} - {MAX_HOURS})."
        )
        st.session_state.last_submission = None
    else:
        st.session_state.last_submission = {
            "name": name.strip(),
            "hours": float(hours),
        }

# ---------------- Result ----------------
submission = st.session_state.last_submission

if submission is not None:
    s_name = submission["name"]
    s_hours = submission["hours"]

    pred = model.predict([[s_hours]])[0]
    prob_pass = model.predict_proba([[s_hours]])[0][1]
    result = "PASS" if pred == 1 else "FAIL"
    confidence = prob_pass if pred == 1 else 1 - prob_pass

    card_class = "result-pass" if pred == 1 else "result-fail"

    st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-label">Prediction for {s_name}</div>
            <p class="result-value">{result}</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Study Hours</div>
                <div class="metric-value">{s_hours}</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Pass Probability</div>
                <div class="metric-value">{prob_pass:.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence:.1%}</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    with st.expander("How was this prediction made?", expanded=True):
        st.write(
            f"The model received **{s_hours} study hours** as input. "
            f"Using patterns learned during training, it calculated a "
            f"**{prob_pass:.1%} probability of passing**. "
            f"Since this is {'above' if pred == 1 else 'below'} the 50% threshold, "
            f"the final classification is **{result}**."
        )

    st.write("")
    st.subheader("Pass Probability by Study Hours")

    hours_range = np.arange(MIN_HOURS, MAX_HOURS + 0.1, 0.25)
    probs = [model.predict_proba([[h]])[0][1] for h in hours_range]

    chart_df = pd.DataFrame({
        "Study Hours": hours_range,
        "Probability": probs,
        "Type": ["Your Input" if abs(h - s_hours) < 0.13 else "Baseline" for h in hours_range],
    })

    try:
        import altair as alt

        base = alt.Chart(chart_df).encode(
            x=alt.X("Study Hours:Q", axis=alt.Axis(title="Study Hours", format=".1f")),
            y=alt.Y("Probability:Q", axis=alt.Axis(format="%", title="Probability of Passing")),
        )

        area = base.mark_area(
            line={"color": "#3b82f6"},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(59,130,246,0.05)", offset=0),
                    alt.GradientStop(color="rgba(59,130,246,0.5)", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
        )
        line = base.mark_line(color="#3b82f6", strokeWidth=3)
        marker = base.mark_circle(size=200, color="#f59e0b").encode(
            tooltip=["Study Hours", alt.Tooltip("Probability:Q", format=".1%")]
        ).transform_filter(alt.datum.Type == "Your Input")

        threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
            color="#ef4444", strokeDash=[5, 5], strokeWidth=1.5
        ).encode(y="y:Q")

        chart = (area + line + threshold + marker).properties(height=350)
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"Orange dot = your input ({s_hours} hrs). "
            f"Red dashed line = 50% decision threshold."
        )
    except ImportError:
        st.bar_chart(chart_df.set_index("Study Hours")["Probability"])

# ---------------- Footer ----------------
st.divider()
st.caption("IT Elective 3 - AI: Machine Learning | Model Deployment Demo")