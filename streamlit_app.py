import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------- Page config ----------
st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered"
)

# ============================================================
# CUSTOM CSS — Background color sa title page
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #e2e8f0;
    }

    /* Title banner with background color */
    .title-banner {
        background: linear-gradient(90deg, #dc2626 0%, #f97316 50%, #eab308 100%);
        padding: 35px 40px;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(220, 38, 38, 0.4);
    }
    .title-banner h1 {
        color: #ffffff;
        font-size: 2.6rem;
        margin: 0;
        font-weight: 800;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
    }
    .title-banner p {
        color: #fff7ed;
        font-size: 1.05rem;
        margin-top: 10px;
        margin-bottom: 0;
        letter-spacing: 1px;
        font-weight: 500;
    }

    label, .stMarkdown p, .stMarkdown li {
        color: #e2e8f0 !important;
    }
    h1, h2, h3 {
        color: #fbbf24 !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #dc2626, #f97316);
        color: white;
        border: none;
        padding: 14px 32px;
        font-size: 1.15rem;
        font-weight: 700;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(220, 38, 38, 0.7);
    }

    .result-pass {
        background: linear-gradient(90deg, #16a34a, #22c55e);
        color: white;
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 800;
        box-shadow: 0 6px 20px rgba(22, 163, 74, 0.5);
        margin: 15px 0;
    }
    .result-fail {
        background: linear-gradient(90deg, #dc2626, #ef4444);
        color: white;
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 800;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.5);
        margin: 15px 0;
    }

    .streamlit-expanderHeader {
        color: #fbbf24 !important;
        font-weight: 600;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

# ============================================================
# TITLE BANNER
# ============================================================
st.markdown("""
<div class="title-banner">
    <h1>🎓 Student Pass Predictor</h1>
    <p>Enter a student's name and study hours to see if they're likely to pass.</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter the student's details below and click **Predict** to estimate "
    "whether they will pass the course."
)

st.divider()

# ---------- Inputs ----------
student_name = st.text_input("Student name")

study_hours = st.slider(
    "Study hours",
    min_value=0.0, max_value=24.0, value=5.0, step=0.5
)

st.divider()

# ---------- Prediction button ----------
if st.button("🔮 Predict", type="primary", use_container_width=True):

    if not student_name.strip():
        st.warning("⚠️ Please enter the student's name before predicting.")
        st.stop()

    X_input = np.array([[study_hours]])
    prediction = int(model.predict(X_input)[0])
    probability = float(model.predict_proba(X_input)[0][1])

    st.subheader(f"📋 Result for {student_name.strip()}")

    if prediction == 1:
        st.markdown(
            f'<div class="result-pass">✅ PREDICTED: PASS<br>'
            f'<span style="font-size:1rem;">Pass probability: {probability:.1%}</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-fail">❌ PREDICTED: FAIL<br>'
            f'<span style="font-size:1rem;">Pass probability: {probability:.1%}</span></div>',
            unsafe_allow_html=True
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("📚 Study Hours", f"{study_hours} h")
    col2.metric("🎯 Prediction", "PASS" if prediction == 1 else "FAIL")
    col3.metric("📊 Probability", f"{probability:.1%}")

    with st.expander("ℹ️ How was this prediction generated?", expanded=False):
        st.markdown(f"""
        **Step-by-step explanation**

        1. **Input collected:** the model received `study_hours = {study_hours}`.
        2. **Model used:** a *Logistic Regression* classifier.
        3. **Computation:** `z = w · x + b`, then `p = 1 / (1 + e^(−z))`.
        4. **Decision rule:** if `p ≥ 0.5` → **PASS**, else → **FAIL**.
        5. **This prediction's probability:** `p = {probability:.4f}` →
           classified as **{"PASS" if prediction == 1 else "FAIL"}**.
        """)

    # ============================================================
    # GRAPH — 🟢 GREEN = PASS, 🔴 RED = FAIL
    # ============================================================
    st.subheader("📊 Study Hours vs. Pass Probability")

    hours_range = np.linspace(0, 24, 120).reshape(-1, 1)
    probs = model.predict_proba(hours_range)[:, 1]
    classes = (probs >= 0.5).astype(int)

    chart_df = pd.DataFrame({
        "Study Hours": hours_range.flatten(),
        "Pass Probability": probs,
        "Predicted Class": classes,
    })

    fig = go.Figure()

    # Probability curve
    fig.add_trace(go.Scatter(
        x=chart_df["Study Hours"],
        y=chart_df["Pass Probability"],
        mode="lines",
        name="Pass Probability Curve",
        line=dict(color="#fbbf24", width=3),
        hovertemplate="Study Hours: %{x:.1f}<br>Probability: %{y:.3f}<extra></extra>"
    ))

    # 🟢 GREEN POINTS for PASS predictions
    pass_points = chart_df[chart_df["Predicted Class"] == 1]
    fig.add_trace(go.Scatter(
        x=pass_points["Study Hours"],
        y=pass_points["Pass Probability"],
        mode="markers",
        name="PASS",
        marker=dict(
            size=9,
            color="green",
            symbol="circle",
            line=dict(color="darkgreen", width=1)
        ),
        hovertemplate="<b>PASS</b><br>Study Hours: %{x:.1f}<br>Probability: %{y:.3f}<extra></extra>"
    ))

    # 🔴 RED POINTS for FAIL predictions
    fail_points = chart_df[chart_df["Predicted Class"] == 0]
    fig.add_trace(go.Scatter(
        x=fail_points["Study Hours"],
        y=fail_points["Pass Probability"],
        mode="markers",
        name="FAIL",
        marker=dict(
            size=9,
            color="red",
            symbol="circle",
            line=dict(color="darkred", width=1)
        ),
        hovertemplate="<b>FAIL</b><br>Study Hours: %{x:.1f}<br>Probability: %{y:.3f}<extra></extra>"
    ))

    # 🟡 User input point (gold circle with white border)
    fig.add_trace(go.Scatter(
        x=[study_hours],
        y=[probability],
        mode="markers",
        name="Your Input",
        marker=dict(
            size=20,
            color="gold",
            symbol="circle",
            line=dict(color="white", width=3)
        ),
        hovertemplate=f"<b>Your Input</b><br>Study Hours: {study_hours}<br>Probability: {probability:.3f}<extra></extra>"
    ))

    # Threshold line at 0.5
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.6)",
        annotation_text="Decision Threshold (0.5)",
        annotation_position="top right",
        annotation_font_color="#e2e8f0"
    )

    fig.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        paper_bgcolor="rgba(30, 41, 59, 0.9)",
        font=dict(color="#e2e8f0", size=13),
        xaxis=dict(
            title="Study Hours",
            gridcolor="rgba(148, 163, 184, 0.2)",
            zerolinecolor="rgba(148, 163, 184, 0.3)"
        ),
        yaxis=dict(
            title="Pass Probability",
            gridcolor="rgba(148, 163, 184, 0.2)",
            zerolinecolor="rgba(148, 163, 184, 0.3)",
            range=[-0.05, 1.05]
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(220, 38, 38, 0.6)",
            borderwidth=1
        ),
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "🟢 **Green points** = predicted PASS · "
        "🔴 **Red points** = predicted FAIL · "
        "🟡 **Gold point** = your input. "
        f"Your input ({study_hours} h) has p = {probability:.2f}."
    )

else:
    st.info("💡 Fill in the details above and press **Predict** to continue.")