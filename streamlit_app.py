# Exercise 3 - Streamlit interface (study hours + attendance) with colored dashboard design
import streamlit as st
import joblib
import pandas as pd

# ---------- Page setup ----------
st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

st.markdown(
    """
    <style>
    /* Page background */
    .stApp {background: linear-gradient(180deg, #eef2ff 0%, #f8fafc 60%);}

    /* Hero header (gradient banner) */
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #06b6d4 100%);
        padding: 2rem 1.5rem; border-radius: 18px; text-align: center;
        color: #ffffff; margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.30);
    }
    .hero h1 {color:#ffffff; font-size:2.2rem; margin:0; padding:0;}
    .hero p  {color:#e0e7ff; margin:0.4rem 0 0 0; font-size:1.05rem;}

    /* Form card */
    div[data-testid="stForm"] {
        background: #ffffff; border: 1px solid #e0e7ff; border-radius: 16px;
        padding: 1.5rem; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    /* Predict button */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: #ffffff; border: none; border-radius: 10px;
        font-weight: 600; padding: 0.6rem 1rem;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #4338ca, #6d28d9); color: #ffffff;
    }

    /* Result cards */
    .result-card {
        padding: 1.3rem 1.5rem; border-radius: 14px; margin: 1.2rem 0;
        font-size: 1.3rem; font-weight: 700; text-align: center;
    }
    .pass {background: linear-gradient(135deg, #d1fae5, #a7f3d0); color:#065f46; border:1px solid #6ee7b7;}
    .fail {background: linear-gradient(135deg, #fee2e2, #fecaca); color:#991b1b; border:1px solid #fca5a5;}

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #ffffff; border-left: 5px solid #4f46e5; border-radius: 12px;
        padding: 0.9rem 1rem; box-shadow: 0 3px 10px rgba(15, 23, 42, 0.07);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {background: #e0e7ff;}

    /* Section headings */
    h2, h3 {color: #3730a3;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model_v2.joblib")


model = load_model()

# ---------- Hero header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Pass Predictor</h1>
        <p>Enter the student details to predict Pass or Fail</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a Logistic Regression model trained on study hours "
        "and attendance percentage."
    )
    st.caption("Model file: student_pass_model_v2.joblib")

# ---------- Input form ----------
with st.form("prediction_form"):
    name = st.text_input("👤 Student name", placeholder="e.g. Juan Dela Cruz")

    col1, col2 = st.columns(2)
    with col1:
        hours = st.number_input(
            "📚 Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5
        )
    with col2:
        attendance = st.number_input(
            "🗓️ Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0
        )

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        features = [[hours, attendance]]
        pred = int(model.predict(features)[0])
        prob = float(model.predict_proba(features)[0][1])

        # Result message
        if pred == 1:
            st.markdown(
                f'<div class="result-card pass">✅ {name} is predicted to PASS</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-card fail">❌ {name} is predicted to FAIL</div>',
                unsafe_allow_html=True,
            )

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Study hours", f"{hours:g}")
        m2.metric("Attendance", f"{attendance:g}%")
        m3.metric("Pass probability", f"{prob:.0%}")
        st.progress(prob)

        # Short explanation
        with st.expander("ℹ️ How was this prediction generated?", expanded=True):
            st.write(
                "The model was trained on past examples of study hours, attendance "
                "percentage, and pass/fail results. It combines the two values you "
                "entered to estimate the probability of passing. If that probability "
                f"is above 50%, it predicts **Pass**. For {name}, the estimated "
                f"probability is **{prob:.0%}**, so the prediction is "
                f"**{'Pass' if pred == 1 else 'Fail'}**."
            )

        # Bonus chart: study hours vs predicted class (attendance kept at the entered value)
        st.subheader("📈 Study hours vs predicted class")
        hours_range = [h / 2 for h in range(0, 25)]
        sample = [[h, attendance] for h in hours_range]
        chart_df = pd.DataFrame(
            {
                "Study hours": hours_range,
                "Predicted class (1=Pass, 0=Fail)": model.predict(sample),
                "Pass probability": model.predict_proba(sample)[:, 1],
            }
        ).set_index("Study hours")

        st.line_chart(chart_df, color=["#7c3aed", "#06b6d4"])
        st.caption(
            f"Chart uses attendance = {attendance:g}%. "
            f"{name}'s input: {hours:g} hours → class {pred}."
        )