
from pathlib import Path

import joblib
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = Path("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

st.markdown(
    """
    <style>
        :root {
            --sky: #dff9ff;
            --cloud: #ffffff;
            --sun: #ffd75f;
            --pink: #ffafd2;
            --purple: #c8b6ff;
            --green: #9ce8b1;
            --red: #ff9ca3;
            --text: #5e3d3a;
            --muted: #7f5d5a;
            --card: #fffdf8;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(255,255,255,0.9) 0 13%, transparent 14%),
                radial-gradient(circle at 80% 18%, rgba(255,255,255,0.9) 0 11%, transparent 12%),
                radial-gradient(circle at 20% 70%, rgba(255,255,255,0.9) 0 12%, transparent 13%),
                linear-gradient(180deg, #eafcff 0%, #fff8d6 38%, #ffe4f2 100%);
            color: var(--text);
            font-family: "Comic Sans MS", "Trebuchet MS", "Segoe Print", sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-shell {
            background: linear-gradient(135deg, #fffef8 0%, #fff0b4 100%);
            border: 4px solid #f7c94d;
            border-radius: 30px;
            padding: 1.5rem 1.3rem;
            box-shadow: 0 10px 0 rgba(247, 201, 77, 0.45), 0 20px 35px rgba(255, 174, 218, 0.18);
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
        }

        .app-shell::before,
        .app-shell::after {
            content: "";
            position: absolute;
            width: 100px;
            height: 100px;
            background: rgba(255,255,255,0.5);
            border-radius: 50%;
            top: -25px;
        }

        .app-shell::before { left: 15%; }
        .app-shell::after { right: 12%; }

        h1 {
            color: #ff7e6a !important;
            font-size: 2.7rem !important;
            text-align: center;
            font-weight: 900 !important;
            letter-spacing: 0.03em;
            margin-bottom: 0.2rem !important;
            text-shadow: 2px 2px 0 rgba(255,255,255,0.8);
        }

        .subtitle {
            color: #7d5c57;
            font-size: 1.08rem;
            text-align: center;
            font-weight: 800;
        }

        .info-card {
            background: linear-gradient(135deg, #fffdf8 0%, #eefbff 100%);
            border: 4px solid #ffd166;
            border-radius: 28px;
            padding: 1.2rem;
            box-shadow: 0 10px 0 rgba(255, 209, 102, 0.35);
        }

        .result-box {
            border-radius: 22px;
            padding: 1rem 1.2rem;
            margin-top: 1.2rem;
            font-size: 1.15rem;
            font-weight: 900;
            text-align: center;
            border: 3px solid rgba(255,255,255,0.6);
            box-shadow: inset 0 0 0 2px rgba(255,255,255,0.35), 0 8px 18px rgba(0,0,0,0.06);
        }

        .result-pass {
            background: linear-gradient(135deg, #c7f9cc 0%, #a8f5d1 100%);
            color: #0f5132;
        }

        .result-fail {
            background: linear-gradient(135deg, #ffd4d4 0%, #f9d0ff 100%);
            color: #7f1d1d;
        }

        .summary-card {
            background: linear-gradient(135deg, #fffaf7 0%, #edfaff 100%);
            border: 3px solid #ffd166;
            border-radius: 22px;
            padding: 1rem 1.1rem;
            margin-top: 1rem;
            box-shadow: 0 8px 0 rgba(255, 209, 102, 0.3);
        }

        .summary-title {
            font-size: 1.2rem;
            font-weight: 900;
            color: #7c4f9d;
            margin-bottom: 0.8rem;
            text-align: center;
        }

        .summary-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            padding: 0.55rem 0.75rem;
            border-radius: 14px;
            margin-bottom: 0.5rem;
            background: rgba(255,255,255,0.55);
            color: #4d3a3c;
            font-weight: 700;
        }

        .summary-row strong {
            color: #2d6a4f;
            text-align: right;
        }

        .summary-row.highlight {
            background: linear-gradient(135deg, #d8fbe0 0%, #c3ecff 100%);
            border: 2px solid rgba(101, 173, 255, 0.4);
        }

        .summary-row.highlight strong {
            color: #0f766e;
        }

        .stButton > button {
            background: linear-gradient(135deg, #7cc8ff 0%, #ff9cc8 100%);
            color: white;
            border: none;
            border-radius: 18px;
            font-weight: 900;
            font-size: 1.05rem;
            padding: 0.8rem 1.4rem;
            box-shadow: 0 8px 0 rgba(255, 156, 200, 0.4);
        }

        .stButton > button:hover {
            filter: brightness(1.06);
            transform: translateY(-1px);
        }

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background: #ffffff;
            color: #4d3a3c;
            border: 2px solid #ffbfd9;
            border-radius: 14px;
            font-weight: 700;
            font-family: "Comic Sans MS", "Trebuchet MS", "Segoe Print", sans-serif;
        }

        .stTextInput label,
        .stNumberInput label,
        .stMarkdown p,
        .stMarkdown li,
        .stMarkdown div {
            color: #4d3a3c !important;
        }

        .stAlert {
            border-radius: 16px;
            border: 2px solid rgba(0,0,0,0.05);
            background: #fffdf6;
            color: #5e3d3a;
        }

        .stMarkdown h3 {
            color: #7a5ada !important;
            font-weight: 800 !important;
            font-family: "Comic Sans MS", "Trebuchet MS", "Segoe Print", sans-serif;
        }

        .stDataFrame, .stChart {
            border-radius: 20px;
            overflow: hidden;
        }

        .chart-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-family: "Comic Sans MS", "Trebuchet MS", "Segoe Print", sans-serif;
            font-size: 2.2rem;
            font-weight: 900;
            color: #7b5ee9;
            margin: 0.5rem 0 1rem 0;
            text-shadow: 2px 2px 0 rgba(255,255,255,0.7);
        }

        .chart-panel {
            background: #131c22;
            border-radius: 18px;
            padding: 0.75rem 0.75rem 0.25rem 0.75rem;
            border: 3px solid rgba(255,255,255,0.18);
            box-shadow: inset 0 0 0 2px rgba(255,255,255,0.04);
        }

        .chart-legend {
            color: #f3e9d7; 
            font-size: 0.95rem; 
            font-weight: 700; 
            margin-top: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-shell">
        <h1>🌈 Study Buddy Predictor</h1>
        <div class="subtitle">Let’s see if our learner is ready to shine! ✨</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.warning("Model file not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.markdown("<div class='info-card'>", unsafe_allow_html=True)
left_col, right_col = st.columns([1.3, 1])

with left_col:
    name = st.text_input("Student Name", placeholder="e.g. Maria Santos")
    hours = st.number_input("Study Hours", min_value=0.0, step=0.5, help="Total hours studied")
    attendance = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        help="Class attendance in percent",
    )
    predict = st.button("Predict Outcome", use_container_width=True)

with right_col:
    st.markdown("### Model Summary")
    st.markdown("- Uses a trained logistic regression classifier")
    st.markdown("- Predicts Pass or Fail from study inputs")
    st.markdown("- Designed for quick student performance evaluation")

st.markdown("</div>", unsafe_allow_html=True)

if predict:
    if not name:
        st.warning("Please enter the student's name.")
        st.stop()

    if attendance < 0 or attendance > 100:
        st.error("Attendance must be between 0 and 100.")
        st.stop()

    result = model.predict([[hours]])
    prediction = "Pass" if result[0] == 1 else "Fail"

    if prediction == "Pass":
        st.markdown(
            f"<div class='result-box result-pass'>{name} is likely to <strong>Pass</strong>.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='result-box result-fail'>{name} is likely to <strong>Fail</strong>.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-title">📘 Student Result</div>
            <div class="summary-row"><span>Student Name</span><strong>{name}</strong></div>
            <div class="summary-row"><span>Study Hours</span><strong>{hours} hours</strong></div>
            <div class="summary-row"><span>Attendance</span><strong>{attendance}%</strong></div>
            <div class="summary-row highlight"><span>Prediction</span><strong>{prediction}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    explanation = (
        "This result was generated by the trained machine learning model using the entered study hours "
        "and the current attendance values."
    )
    st.info(explanation)

    x_axis = [0, 1, 2, 3]
    study_series = [0, max(hours / 2, 0.5), hours, max(hours, 1)]
    prediction_series = [0, 0.5 if prediction == "Pass" else 0.2, 1 if prediction == "Pass" else 0.2, 1 if prediction == "Pass" else 0.2]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=study_series,
            mode="lines+markers",
            name="Study Hours",
            line=dict(color="#8ad9b2", width=3),
            marker=dict(size=8, color="#8ad9b2"),
            fill="tozeroy",
            fillcolor="rgba(138, 217, 178, 0.18)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=prediction_series,
            mode="lines+markers",
            name="Prediction",
            line=dict(color="#c8b6ff", width=3),
            marker=dict(size=8, color="#c8b6ff"),
            fill="tozeroy",
            fillcolor="rgba(200, 182, 255, 0.14)",
        )
    )

    fig.update_layout(
        paper_bgcolor="#0d1b22",
        plot_bgcolor="#0d1b22",
        margin=dict(l=30, r=20, t=10, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb", size=13),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(color="#dfe7ec"),
            title="",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(color="#dfe7ec"),
            title="",
        ),
        font=dict(color="#f5f5f5", family="Segoe UI, sans-serif"),
        hovermode="x unified",
    )

    st.markdown(
        """
        <div class="chart-header">
            <span>📊</span>
            <span>Student Progress Chart</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.markdown('</div>', unsafe_allow_html=True)
