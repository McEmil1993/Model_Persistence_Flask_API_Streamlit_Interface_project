"""Exercise 3 - Streamlit interface for the student pass model (redesigned)."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 820px; }
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        padding: 2rem 2rem 1.6rem 2rem; border-radius: 18px; color: white;
        margin-bottom: 1.5rem; box-shadow: 0 10px 30px rgba(79,70,229,.25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p { margin: .4rem 0 0 0; opacity: .92; font-size: 1.02rem; }
    .result-card {
        padding: 1.4rem 1.6rem; border-radius: 16px; margin: 1.2rem 0 .8rem 0;
        border: 1px solid rgba(128,128,128,.25);
    }
    .pass { background: rgba(34,197,94,.12); border-left: 8px solid #22c55e; }
    .fail { background: rgba(239,68,68,.12); border-left: 8px solid #ef4444; }
    .result-card h2 { margin: 0 0 .2rem 0; font-size: 1.6rem; }
    .result-card p { margin: 0; opacity: .85; }
    div.stButton > button {
        width: 100%; border-radius: 12px; padding: .7rem 1rem; font-weight: 600;
        border: none; color: white;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
    }
    div.stButton > button:hover { filter: brightness(1.1); color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ How it works")
    st.write(
        "A **logistic regression** model was trained once on study-hour data and "
        "saved with `joblib`. This app only **loads** that file, so no retraining "
        "happens when you click Predict."
    )
    st.write("Pass probability **≥ 50%** → PASS, otherwise FAIL.")

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Pass Predictor</h1>
        <p>Enter a student's daily study hours and see the predicted outcome.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Inputs ----------
col1, col2 = st.columns([1, 1])
with col1:
    name = st.text_input("Student name", placeholder="e.g. Maria")
with col2:
    hours = st.number_input("Study hours per day", min_value=0.0, max_value=12.0,
                            value=4.0, step=0.5)

predict_clicked = st.button("🔮 Predict")

# ---------- Result ----------
if predict_clicked:
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])
        label = "PASS" if pred == 1 else "FAIL"
        css = "pass" if pred == 1 else "fail"
        icon = "✅" if pred == 1 else "⚠️"

        st.markdown(
            f"""
            <div class="result-card {css}">
                <h2>{icon} {name} is predicted to {label}</h2>
                <p>Estimated pass probability: <b>{prob:.0%}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Study hours", f"{hours:g} h")
        m2.metric("Pass probability", f"{prob:.0%}")
        m3.metric("Predicted class", label)

        st.progress(prob, text=f"Pass probability: {prob:.0%}")

        st.info(
            f"**How this was generated:** the saved logistic regression model "
            f"(`student_pass_model.joblib`) took **{hours:g} study hours** and "
            f"calculated a pass probability of **{prob:.2f}**. Since "
            f"{'it is at least' if pred else 'it is below'} 0.50, the class is **{label}**."
        )

        # Bonus chart: study hours vs predicted class
        st.subheader("📈 Study hours vs pass probability")
        xs = np.linspace(0, 12, 100)
        probs = model.predict_proba(xs.reshape(-1, 1))[:, 1]
        chart_df = pd.DataFrame(
            {
                "Study hours": xs,
                "Pass probability": probs,
                "Threshold (0.50)": 0.5,
            }
        ).set_index("Study hours")
        st.line_chart(chart_df, color=["#7c3aed", "#9ca3af"])
        st.caption(
            f"{name}: {hours:g} hours → probability {prob:.2f} → predicted {label}."
        )