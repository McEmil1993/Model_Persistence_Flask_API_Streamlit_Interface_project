"""
streamlit_app.py
----------------
IMPROVED UI version:
- Custom background color & title styling
- Red points sa graph
- Walay balloon effect
"""

import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# CUSTOM CSS — BACKGROUND COLOR SA TITLE PAGE
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #e2e8f0;
    }

    /* Title banner */
    .title-banner {
        background: linear-gradient(90deg, #dc2626 0%, #f97316 50%, #eab308 100%);
        padding: 30px 40px;
        border-radius: 16px;
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
        font-size: 1.1rem;
        margin-top: 8px;
    }

    /* Input labels */
    label, .stMarkdown p {
        color: #e2e8f0 !important;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #dc2626, #f97316);
        color: white;
        border: none;
        padding: 12px 32px;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(220, 38, 38, 0.7);
    }

    /* Result cards */
    .result-pass {
        background: linear-gradient(90deg, #16a34a, #22c55e);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        box-shadow: 0 6px 20px rgba(22, 163, 74, 0.5);
    }
    .result-fail {
        background: linear-gradient(90deg, #dc2626, #ef4444);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.5);
    }

    /* Subheaders */
    h2, h3 {
        color: #fbbf24 !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE BANNER (with background color)
# ============================================================
st.markdown("""
<div class="title-banner">
    <h1>🎓 Student Performance Prediction</h1>
    <p>Machine Learning Model · Flask API · Streamlit Interface</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
model = joblib.load("student_model.joblib")

# ============================================================
# INPUT SECTION
# ============================================================
st.subheader("📝 Enter Student Data")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "📚 Study Hours",
        min_value=0.0, max_value=24.0, value=5.0, step=0.5
    )

with col2:
    attendance = st.number_input(
        "📅 Attendance (%)",
        min_value=0.0, max_value=100.0, value=75.0, step=1.0
    )

# ============================================================
# PREDICT BUTTON
# ============================================================
if st.button("🔮 Predict Result"):
    input_df = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["study_hours", "attendance"]
    )
    prediction = model.predict(input_df)[0]
    result = "Pass" if prediction == 1 else "Fail"

    if result == "Pass":
        st.markdown('<div class="result-pass">✅ PREDICTION: PASS</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-fail">❌ PREDICTION: FAIL</div>',
                    unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    c1.metric("Study Hours", f"{study_hours} hrs")
    c2.metric("Attendance", f"{attendance}%")

# ============================================================
# GRAPH — RED POINTS (Scatter Plot)
# ============================================================
st.subheader("📊 Student Performance Graph (Red Points = Students)")

df = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance":  [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "result":      ["Fail","Fail","Fail","Fail","Pass","Pass","Pass","Pass","Pass","Pass"]
})

fig = go.Figure()

# Red points for FAIL students
fail_df = df[df["result"] == "Fail"]
fig.add_trace(go.Scatter(
    x=fail_df["study_hours"],
    y=fail_df["attendance"],
    mode="markers",
    name="Fail",
    marker=dict(
        size=18,
        color="red",
        symbol="circle",
        line=dict(color="darkred", width=2)
    ),
    hovertemplate="<b>Fail</b><br>Study Hours: %{x}<br>Attendance: %{y}%<extra></extra>"
))

# Red points (with different shade) for PASS students
pass_df = df[df["result"] == "Pass"]
fig.add_trace(go.Scatter(
    x=pass_df["study_hours"],
    y=pass_df["attendance"],
    mode="markers",
    name="Pass",
    marker=dict(
        size=18,
        color="red",
        symbol="diamond",
        line=dict(color="darkred", width=2)
    ),
    hovertemplate="<b>Pass</b><br>Study Hours: %{x}<br>Attendance: %{y}%<extra></extra>"
))

# Highlight the user's input point (bigger red star)
if True:
    fig.add_trace(go.Scatter(
        x=[study_hours],
        y=[attendance],
        mode="markers",
        name="Your Input",
        marker=dict(
            size=26,
            color="red",
            symbol="star",
            line=dict(color="white", width=2)
        ),
        hovertemplate="<b>Your Input</b><br>Study Hours: %{x}<br>Attendance: %{y}%<extra></extra>"
    ))

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
        title="Attendance (%)",
        gridcolor="rgba(148, 163, 184, 0.2)",
        zerolinecolor="rgba(148, 163, 184, 0.3)"
    ),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.8)",
        bordercolor="rgba(220, 38, 38, 0.6)",
        borderwidth=1
    ),
    height=500,
    margin=dict(l=40, r=40, t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DATA TABLE
# ============================================================
st.subheader("📋 Training Data")
st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================================
# COMPARISON TABLE (Flask vs Streamlit)
# ============================================================
with st.expander("ℹ️ Flask API vs Streamlit"):
    st.markdown("""
| Feature | Flask API | Streamlit |
|---|---|---|
| Main purpose | Provides predictions through API endpoints | Builds interactive web applications |
| User interface | Usually built separately | Built directly using Python |
| Response | Often JSON | Visual interface with displayed results |
| Common users | Developers integrating systems | Data scientists, students, and model demonstrators |
| Best for | Backend services and integrations | Prototypes, demos, and ML dashboards |
    """)