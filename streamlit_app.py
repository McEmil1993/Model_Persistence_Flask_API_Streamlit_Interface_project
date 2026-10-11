"""Student Pass Prediction System - polished Streamlit interface."""

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = "http://127.0.0.1:5000/predict"

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# PROFESSIONAL UI STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #856C74;
        --line: #E6EAF0;
        --blue: #E85D75;
        --blue-dark: #C83E67;
        --surface: #FFFFFF;
        --canvas: #F6F8FC;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(232, 93, 117, 0.08), transparent 32%),
            var(--canvas);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(246, 248, 252, 0.85);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', 'DM Sans', sans-serif !important;
        color: var(--ink);
        letter-spacing: -0.035em;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 28px 30px;
        border-radius: 22px;
        background: linear-gradient(120deg, #C83E67 0%, #E85D75 55%, #F59A8B 100%);
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 14px 34px rgba(200, 62, 103, 0.20);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 50%;
        right: -65px;
        top: -105px;
        box-shadow: 0 0 0 28px rgba(255,255,255,.04),
                    0 0 0 58px rgba(255,255,255,.035);
    }

    .hero-eyebrow {
        font-size: 11px;
        letter-spacing: .16em;
        text-transform: uppercase;
        font-weight: 700;
        color: #FFE5EC;
        margin-bottom: 9px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(25px, 3.2vw, 36px);
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -.045em;
        color: #FFFFFF;
        margin-bottom: 9px;
        position: relative;
        z-index: 1;
    }

    .hero-copy {
        color: #FFF0F3;
        font-size: 14px;
        max-width: 610px;
        line-height: 1.65;
        position: relative;
        z-index: 1;
        margin-bottom: 0;
    }

    .section-kicker {
        color: #C83E67;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: 10px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .section-help {
        color: var(--muted);
        font-size: 13px;
        margin-top: -7px;
        margin-bottom: 18px;
    }

    div[data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 22px 22px 16px;
        box-shadow: 0 5px 18px rgba(23, 32, 51, 0.035);
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        color: #344054;
        font-size: 13px;
        font-weight: 600;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 11px;
        min-height: 44px;
        font-weight: 700;
        border: 1px solid #E85D75;
        background: linear-gradient(100deg, #D9486B, #F0788D);
        box-shadow: 0 5px 12px rgba(232, 93, 117, 0.18);
        transition: transform .15s ease, box-shadow .15s ease;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        border-color: #C83E67;
        box-shadow: 0 8px 18px rgba(232, 93, 117, 0.25);
        transform: translateY(-1px);
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 17px 18px;
        box-shadow: 0 5px 18px rgba(23, 32, 51, 0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: #856C74;
        font-size: 12px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #172033;
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
    }

    .result-card {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 20px 22px;
        margin: 8px 0 18px;
        box-shadow: 0 5px 18px rgba(23, 32, 51, 0.035);
    }

    .result-label {
        color: #856C74;
        text-transform: uppercase;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .13em;
        margin-bottom: 8px;
    }

    .result-pass {
        color: #13845D;
        font-family: 'Manrope', sans-serif;
        font-size: 27px;
        font-weight: 800;
    }

    .result-fail {
        color: #D9485F;
        font-family: 'Manrope', sans-serif;
        font-size: 27px;
        font-weight: 800;
    }

    .result-detail {
        color: #856C74;
        font-size: 13px;
        margin-top: 5px;
    }

    .info-panel {
        background: linear-gradient(135deg, #FFFFFF 0%, #F7F8FF 100%);
        border: 1px solid #F8D7E0;
        border-radius: 17px;
        padding: 20px 22px;
        line-height: 1.75;
        color: #6F4A55;
        font-size: 14px;
    }

    .info-panel strong {
        color: #71384B;
    }

    .chart-footnote {
        color: #856C74;
        font-size: 12px;
        line-height: 1.6;
        margin-top: -5px;
    }

    .footer {
        color: #8993A4;
        font-size: 11px;
        text-align: center;
        padding: 8px 0 0;
    }

    hr {
        border-color: #E6EAF0 !important;
        margin-top: 26px !important;
        margin-bottom: 22px !important;
    }

    @media (max-width: 640px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 22px 20px; border-radius: 17px; }
        div[data-testid="stForm"] { padding: 16px 14px 12px; }
        .result-card { padding: 17px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <section class="hero">
        <div class="hero-eyebrow">Student analytics · Prediction dashboard</div>
        <div class="hero-title">Understand performance.<br>Plan for progress.</div>
        <p class="hero-copy">
            Estimate a student's predicted outcome using study hours and
            attendance, then explore how the model's prediction changes
            across different study-hour values.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
st.markdown('<div class="section-kicker">01 / Student details</div>', unsafe_allow_html=True)
st.subheader("Enter student information")
st.markdown(
    '<p class="section-help">Complete the fields below to generate a prediction.</p>',
    unsafe_allow_html=True,
)

with st.form("student_prediction_form"):
    student_name = st.text_input(
        "Student name",
        placeholder="e.g., Gellan D. Golosino",
    )

    input_left, input_right = st.columns(2, gap="large")

    with input_left:
        study_hours = st.number_input(
            "Study hours per day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
            help="Enter the average number of hours spent studying per day.",
        )

    with input_right:
        attendance = st.slider(
            "Attendance percentage",
            min_value=0,
            max_value=100,
            value=85,
            step=1,
            help="Select the student's attendance percentage.",
        )

    submitted = st.form_submit_button(
        "Generate prediction  →",
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if submitted:
    if not student_name.strip():
        st.error("Please enter the student's name before continuing.")
    else:
        payload = {
            "student_name": student_name.strip(),
            "study_hours": float(study_hours),
            "attendance_percentage": float(attendance),
        }

        try:
            with st.spinner("Analyzing student information..."):
                response = requests.post(API_URL, json=payload, timeout=15)
                response.raise_for_status()
                result = response.json()

            prediction = str(result["prediction"]).strip()
            normalized_prediction = prediction.lower()

            if normalized_prediction not in ("pass", "fail"):
                st.error(f"The API returned an unexpected class label: {prediction}")
                st.stop()

            display_prediction = "Pass" if normalized_prediction == "pass" else "Fail"

            st.divider()
            st.markdown('<div class="section-kicker">02 / Prediction summary</div>', unsafe_allow_html=True)
            st.subheader("Your prediction")

            result_class = "result-pass" if display_prediction == "Pass" else "result-fail"
            result_symbol = "✓" if display_prediction == "Pass" else "!"
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted outcome · {student_name.strip()}</div>
                    <div class="{result_class}">{result_symbol}&nbsp; {display_prediction}</div>
                    <div class="result-detail">
                        Model-generated estimate based on the submitted student information.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric1, metric2, metric3 = st.columns(3, gap="medium")
            metric1.metric("Study hours", f"{float(result['study_hours']):g} hrs/day")
            metric2.metric("Attendance", f"{float(result['attendance_percentage']):g}%")
            metric3.metric("Predicted class", display_prediction)

            if result.get("message"):
                st.caption(result["message"])

            # -------------------------------------------------
            # CHART DATA: ACTUAL FLASK API PREDICTIONS
            # -------------------------------------------------
            st.divider()
            st.markdown('<div class="section-kicker">03 / Performance insights</div>', unsafe_allow_html=True)
            st.subheader("Study hours vs. predicted class")
            st.markdown(
                '<p class="section-help">'
                'Attendance stays fixed while study hours change. Each point is a prediction returned by your model.'
                '</p>',
                unsafe_allow_html=True,
            )

            hours_to_test = sorted(
                set([float(h) for h in range(0, 25, 4)] + [float(study_hours)])
            )
            chart_rows = []

            with st.spinner("Building the performance chart..."):
                for hours in hours_to_test:
                    chart_payload = {
                        "student_name": student_name.strip(),
                        "study_hours": hours,
                        "attendance_percentage": float(attendance),
                    }
                    chart_response = requests.post(
                        API_URL,
                        json=chart_payload,
                        timeout=15,
                    )
                    chart_response.raise_for_status()
                    chart_result = chart_response.json()

                    class_name = str(chart_result["prediction"]).strip().lower()
                    if class_name not in ("pass", "fail"):
                        raise ValueError(
                            f"Unexpected class label from API: {chart_result['prediction']}"
                        )

                    chart_rows.append(
                        {
                            "Study Hours": hours,
                            "Predicted Class": "Pass" if class_name == "pass" else "Fail",
                            "Class Level": 1 if class_name == "pass" else 0,
                        }
                    )

            chart_data = pd.DataFrame(chart_rows)

            fig = go.Figure()

            # Soft blue line with polished circular points.
            fig.add_trace(
                go.Scatter(
                    x=chart_data["Study Hours"],
                    y=chart_data["Class Level"],
                    mode="lines+markers",
                    name="Model prediction",
                    line=dict(color="#E85D75", width=3.5),
                    marker=dict(
                        size=10,
                        color="#E85D75",
                        line=dict(color="#FFFFFF", width=2),
                    ),
                    customdata=chart_data[["Predicted Class"]].to_numpy(),
                    hovertemplate=(
                        "<b>Study hours:</b> %{x:g}<br>"
                        "<b>Predicted class:</b> %{customdata[0]}"
                        "<extra></extra>"
                    ),
                )
            )

            # Distinct accent marker for the submitted student.
            current_level = 1 if normalized_prediction == "pass" else 0
            fig.add_trace(
                go.Scatter(
                    x=[float(study_hours)],
                    y=[current_level],
                    mode="markers",
                    name="Current student",
                    marker=dict(
                        size=17,
                        color="#F59E0B",
                        line=dict(color="#FFFFFF", width=3),
                    ),
                    hovertemplate=(
                        "<b>Current student</b><br>"
                        "Study hours: %{x:g}<br>"
                        f"Predicted class: {display_prediction}"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                template="plotly_white",
                height=370,
                margin=dict(l=12, r=15, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans, Arial, sans-serif", size=12, color="#856C74"),
                hovermode="closest",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=11),
                ),
                xaxis=dict(
                    title=dict(text="Study hours per day", font=dict(size=13, color="#6F4A55")),
                    range=[-0.5, 24.5],
                    tickmode="linear",
                    tick0=0,
                    dtick=4,
                    showgrid=True,
                    gridcolor="#F5DDE3",
                    griddash="dash",
                    zeroline=False,
                    showline=False,
                    ticks="outside",
                    tickcolor="#E8C7D0",
                ),
                yaxis=dict(
                    title=None,
                    range=[-0.18, 1.18],
                    tickmode="array",
                    tickvals=[0, 1],
                    ticktext=["Fail", "Pass"],
                    showgrid=True,
                    gridcolor="#F5DDE3",
                    zeroline=False,
                    showline=False,
                    tickfont=dict(size=14, color="#6F4A55"),
                    ticks="",
                ),
                showlegend=True,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

            st.markdown(
                '<p class="chart-footnote">'
                '<span style="color:#E85D75;font-weight:700;">●</span> Deep Rose: model predictions'
                '&nbsp;&nbsp;&nbsp; <span style="color:#F59E0B;font-weight:700;">●</span> Soft Peach: current student'
                '</p>',
                unsafe_allow_html=True,
            )
            st.caption(
                "The line connects predicted class categories; it is not a continuous score or a guarantee "
                "that increasing study hours will cause a student to pass."
            )

            # -------------------------------------------------
            # EXPLANATION
            # -------------------------------------------------
            st.divider()
            st.markdown('<div class="section-kicker">04 / How it works</div>', unsafe_allow_html=True)
            st.subheader("How the prediction was generated")

            # Native Streamlit components avoid displaying raw HTML tags.
            with st.container(border=True):
                st.markdown("**1. Student inputs**")
                st.write(
                    f"The system received **{float(study_hours):g} study hours per day** "
                    f"and **{float(attendance):g}% attendance**."
                )

                st.markdown("---")
                st.markdown("**2. Machine learning analysis**")
                st.write(
                    "Streamlit sent the information to the Flask API. The API passed "
                    "the inputs to the trained model, which uses patterns learned from "
                    "its training data."
                )

                st.markdown("---")
                st.markdown("**3. Predicted outcome**")
                st.write(
                    f"For the submitted information, the model returned **{display_prediction}**. "
                    "This is an estimate, not a guarantee of actual academic performance."
                )


            if result.get("model_note"):
                st.caption(result["model_note"])

            st.caption(
                f"Chart comparison: attendance is held constant at {float(attendance):g}%."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to Flask. Start your API in another terminal, then try again."
            )
        except requests.exceptions.Timeout:
            st.error("The Flask API took too long to respond. Please try again.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"The Flask API returned an error: {exc}")
        except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
            st.error(f"Unable to generate the prediction or chart: {exc}")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown(
    '<div class="footer">'
    'STUDENT PASS PREDICTION SYSTEM &nbsp;·&nbsp; Classroom demonstration only. '
    'This model uses synthetic data and should not be used to make real academic decisions.'
    '</div>',
    unsafe_allow_html=True,
)
