import streamlit as st
import joblib
import pandas as pd
import altair as alt

# Load model once
model = joblib.load("student_pass_model.joblib")

st.set_page_config(
    page_title="AI Student Pass Predictor",
    page_icon="🤖",
    layout="wide"
)

# --- Futuristic AI UI Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 15% 10%, rgba(0, 217, 255, 0.10), transparent 35%),
        radial-gradient(ellipse at 85% 85%, rgba(112, 0, 255, 0.12), transparent 40%),
        #050816;
    color: #e5f7ff;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Typography */
h1, h2, h3, h4 {
    color: #e5f7ff !important;
    font-family: 'Inter', sans-serif !important;
}

p, label {
    color: #b5c8e0;
}

/* Hero header */
.ai-header {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        135deg,
        rgba(0, 217, 255, 0.09),
        rgba(112, 0, 255, 0.10)
    );
    border: 1px solid rgba(0, 217, 255, 0.30);
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 28px;
    box-shadow: 0 0 35px rgba(0, 217, 255, 0.05);
}

.ai-header::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: #00e5ff;
    box-shadow: 0 0 20px #00e5ff;
}

.ai-eyebrow {
    color: #00e5ff;
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    margin-bottom: 12px;
}

.ai-title {
    color: #ffffff;
    font-size: clamp(27px, 4vw, 42px);
    font-weight: 800;
    letter-spacing: -1.2px;
    line-height: 1.2;
}

.ai-subtitle {
    color: #94a9c7;
    font-size: 14px;
    margin-top: 12px;
    line-height: 1.7;
}

.ai-status {
    display: inline-block;
    margin-top: 18px;
    padding: 8px 12px;
    color: #67f7ff;
    background: rgba(0, 229, 255, 0.07);
    border: 1px solid rgba(0, 229, 255, 0.30);
    border-radius: 6px;
    font-family: 'Orbitron', sans-serif;
    font-size: 10px;
    letter-spacing: 1px;
}

/* Dashboard metric cards */
.metric-card {
    background: rgba(12, 23, 48, 0.85);
    border: 1px solid #183653;
    border-radius: 12px;
    padding: 20px;
    min-height: 115px;
    box-shadow: inset 0 0 20px rgba(0, 217, 255, 0.015);
}

.metric-label {
    color: #7895b8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    margin-bottom: 12px;
}

.metric-value {
    color: #e5f7ff;
    font-size: 23px;
    font-weight: 800;
}

.metric-accent {
    color: #00e5ff;
}

/* Panels */
.ai-panel {
    background: rgba(9, 18, 39, 0.92);
    border: 1px solid #183653;
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 22px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
}

.panel-heading {
    color: #e5f7ff;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}

.panel-description {
    color: #7895b8;
    font-size: 12px;
    line-height: 1.7;
    margin-bottom: 22px;
}

/* Input fields */
.stTextInput input,
.stNumberInput input {
    background: #071225 !important;
    color: #f0fbff !important;
    border: 1px solid #204361 !important;
    border-radius: 8px !important;
    min-height: 45px;
}

.stTextInput input::placeholder {
    color: #607895 !important;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 1px rgba(0, 229, 255, 0.30) !important;
}

.stSlider [data-baseweb="slider"] {
    padding-top: 8px;
    padding-bottom: 8px;
}

.stSlider [data-testid="stTickBar"] {
    background: #204361;
}

/* Predict button */
.stButton > button {
    width: 100%;
    min-height: 49px;
    color: #03101d;
    background: linear-gradient(100deg, #00e5ff, #00b8d9);
    border: 1px solid #4af0ff;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.7px;
    transition: all 0.25s ease;
    box-shadow: 0 0 18px rgba(0, 229, 255, 0.10);
}

.stButton > button:hover {
    color: #03101d;
    background: #66f3ff;
    border-color: #a0f8ff;
    box-shadow: 0 0 25px rgba(0, 229, 255, 0.28);
    transform: translateY(-1px);
}

/* Prediction result */
.result-card {
    padding: 28px 20px;
    margin: 18px 0;
    border-radius: 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-pass {
    background: rgba(0, 255, 170, 0.06);
    border: 1px solid rgba(0, 255, 170, 0.40);
    box-shadow: inset 0 0 25px rgba(0, 255, 170, 0.025);
}

.result-fail {
    background: rgba(255, 64, 110, 0.06);
    border: 1px solid rgba(255, 64, 110, 0.40);
    box-shadow: inset 0 0 25px rgba(255, 64, 110, 0.025);
}

.result-caption {
    color: #7895b8;
    font-family: 'Orbitron', sans-serif;
    font-size: 10px;
    letter-spacing: 2px;
}

.result-name {
    color: #dcecff;
    font-size: 16px;
    margin-top: 12px;
}

.result-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 36px;
    font-weight: 800;
    margin-top: 14px;
    text-shadow: 0 0 18px currentColor;
}

/* Info and warning boxes */
.stAlert {
    background: #0b1930;
    border: 1px solid #204361;
    border-radius: 9px;
    color: #dcecff;
}

/* Chart */
div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {
    background: #071225;
    border: 1px solid #183653;
    border-radius: 10px;
    padding: 10px;
}

/* Footer */
.ai-footer {
    text-align: center;
    color: #506987;
    font-family: 'Orbitron', sans-serif;
    font-size: 9px;
    letter-spacing: 2px;
    padding-top: 25px;
}

hr {
    border-color: #183653;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .ai-header {
        padding: 25px 20px;
    }

    .ai-panel {
        padding: 18px;
    }

    .metric-card {
        padding: 14px;
    }

    .metric-value {
        font-size: 18px;
    }

    .result-value {
        font-size: 30px;
    }
}
</style>
""", unsafe_allow_html=True)


# --- AI Header ---
st.markdown("""
<div class="ai-header">
    <div class="ai-eyebrow">ARTIFICIAL INTELLIGENCE SYSTEM</div>
    <div class="ai-title">Student Pass Predictor</div>
    <div class="ai-subtitle">
        Academic performance prediction powered by Machine Learning.
        Analyze study habits and attendance to generate a predicted outcome.
    </div>
    <div class="ai-status">● SYSTEM INTERFACE READY</div>
</div>
""", unsafe_allow_html=True)


# --- Overview Metrics ---
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MODEL ARCHITECTURE</div>
        <div class="metric-value">Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with metric2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">INPUT FEATURES</div>
        <div class="metric-value metric-accent">02</div>
    </div>
    """, unsafe_allow_html=True)

with metric3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">OUTPUT CLASSES</div>
        <div class="metric-value">Pass / Fail</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# --- Main Layout ---
left_col, right_col = st.columns([1, 1.15], gap="large")


with left_col:
    st.markdown("""
    <div class="ai-panel">
        <div class="panel-heading">◈ Student Data Input</div>
        <div class="panel-description">
            Enter the student's information below to initialize prediction.
        </div>
    """, unsafe_allow_html=True)

    student_name = st.text_input(
        "Student Name",
        placeholder="e.g., Juan Dela Cruz"
    )

    study_hours = st.number_input(
        "Study Hours per Week",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )

    attendance = st.slider(
        "Attendance Percentage",
        min_value=0,
        max_value=100,
        value=75
    )

    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "⚡  GENERATE PREDICTION",
        use_container_width=True
    )


with right_col:
    st.markdown("""
    <div class="ai-panel">
        <div class="panel-heading">◈ Prediction Analysis</div>
        <div class="panel-description">
            Machine learning output and academic prediction details.
        </div>
    """, unsafe_allow_html=True)

    if predict_clicked:
        if not student_name.strip():
            st.warning("Please enter the student's name.")
        else:
            # Prediction logic
            pred = model.predict([[study_hours, attendance]])[0]
            label = "Pass" if pred == 1 else "Fail"
            color = "#00ffaa" if pred == 1 else "#ff416c"
            result_class = "result-pass" if pred == 1 else "result-fail"
            symbol = "✓" if pred == 1 else "✕"

            st.markdown(f"""
            <div class="result-card {result_class}">
                <div class="result-caption">PREDICTION OUTPUT</div>
                <div class="result-name">{student_name}</div>
                <div class="result-value" style="color:{color};">
                    {symbol} {label.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info(
                f"The model used a Logistic Regression classifier trained on "
                f"study hours and attendance data. With **{study_hours} study hours** "
                f"and **{attendance}% attendance**, it predicted **{label}**."
            )

            # ==========================================================
            # --- ENHANCED CHART: Study Hours vs Predicted Class ---
            # ==========================================================
            st.markdown("#### 📡 Study Hours vs Predicted Class")

            # Build a range of study hours and predict each one
            hours_range = list(range(0, 13))  # 0 to 12
            predictions = [
                int(model.predict([[h, attendance]])[0])
                for h in hours_range
            ]

            chart_df = pd.DataFrame({
                "Study Hours": hours_range,
                "Predicted Class": predictions,
            })

            # Tag the user's own input so we can highlight it
            chart_df["Input"] = [
                "Your Input" if float(h) == float(study_hours) else "Range"
                for h in chart_df["Study Hours"]
            ]

            # Find decision boundary (first hour where prediction flips to 1)
            pass_hours = [h for h, p in zip(hours_range, predictions) if p == 1]
            boundary_hour = pass_hours[0] if pass_hours else None

            # Base bar chart
            bars = alt.Chart(chart_df).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x=alt.X(
                    "Study Hours:O",
                    title="Study Hours per Week",
                    axis=alt.Axis(labelColor="#b5c8e0", titleColor="#b5c8e0")
                ),
                y=alt.Y(
                    "Predicted Class:Q",
                    title="Predicted Class (1 = Pass, 0 = Fail)",
                    scale=alt.Scale(domain=[0, 1.2]),
                    axis=alt.Axis(labelColor="#b5c8e0", titleColor="#b5c8e0")
                ),
                color=alt.Color(
                    "Input:N",
                    scale=alt.Scale(
                        domain=["Your Input", "Range"],
                        range=["#00ffaa", "#00b8d9"]
                    ),
                    legend=alt.Legend(
                        title=None,
                        labelColor="#b5c8e0",
                        orient="top"
                    )
                ),
                tooltip=[
                    alt.Tooltip("Study Hours:O", title="Study Hours"),
                    alt.Tooltip("Predicted Class:Q", title="Class (1=Pass, 0=Fail)")
                ]
            )

            layers = [bars]

            # Add vertical dashed line at decision boundary
            if boundary_hour is not None:
                boundary_df = pd.DataFrame({"x": [boundary_hour - 0.5]})
                rule = alt.Chart(boundary_df).mark_rule(
                    color="#ff416c",
                    strokeDash=[5, 5],
                    strokeWidth=1.5
                ).encode(x="x:Q")
                layers.append(rule)

            # Add label for boundary
            if boundary_hour is not None:
                label_df = pd.DataFrame({
                    "x": [boundary_hour - 0.5],
                    "y": [1.15],
                    "text": [f"Cutoff ≈ {boundary_hour} hrs"]
                })
                text = alt.Chart(label_df).mark_text(
                    color="#ff416c",
                    fontSize=11,
                    fontWeight="bold",
                    dy=-2
                ).encode(
                    x="x:Q",
                    y="y:Q",
                    text="text:N"
                )
                layers.append(text)

            chart = alt.layer(*layers).properties(
                height=320,
                background="#071225"
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                gridColor="#183653",
                domainColor="#204361"
            )

            st.altair_chart(chart, use_container_width=True)

            st.caption(
                f"Green bar marks your input ({study_hours} study hours). "
                f"Blue bars show what the model predicts across other study-hour values "
                f"(at {attendance}% attendance). "
                + (f"The dashed line marks the pass/fail cutoff at ≈ {boundary_hour} hours."
                   if boundary_hour is not None else "")
            )

    else:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 45px 15px;
            background: #071225;
            border: 1px dashed #204361;
            border-radius: 12px;
        ">
            <div style="
                font-size: 42px;
                margin-bottom: 16px;
                filter: drop-shadow(0 0 12px #00e5ff);
            ">◈</div>
            <div style="
                color: #dcecff;
                font-weight: 700;
                font-size: 15px;
            ">SYSTEM STANDBY</div>
            <div style="
                color: #7895b8;
                font-size: 12px;
                margin-top: 10px;
                line-height: 1.8;
            ">
                Awaiting student data.<br>
                Enter the required details and generate a prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --- Footer ---
st.markdown("""
<div class="ai-footer">
    AI STUDENT ANALYTICS · MACHINE LEARNING INTERFACE
</div>
""", unsafe_allow_html=True)