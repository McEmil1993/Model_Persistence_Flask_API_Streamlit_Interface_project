"""Interactive Streamlit dashboard for predicting student pass/fail outcomes."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_model():
    model_path = Path(__file__).with_name("student_pass_model_v2.joblib")
    if not model_path.exists():
        model_path = Path(__file__).with_name("student_pass_model.joblib")
    return joblib.load(model_path)


model = load_model()


def calculate_prediction(hours, attendance):
    features = pd.DataFrame(
        {"study_hours": [hours], "attendance_percentage": [attendance]}
    )
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    return prediction, probability


def get_probability_curve(attendance):
    hours_range = [hour / 2 for hour in range(0, 25)]
    probabilities = model.predict_proba(
        pd.DataFrame(
            {
                "study_hours": hours_range,
                "attendance_percentage": [attendance] * len(hours_range),
            }
        )
    )[:, 1]
    return pd.DataFrame({"Study hours": hours_range, "Pass probability": probabilities})


st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f7faff 0%, #eef7ff 100%);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.75rem;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #4338ca, #6d28d9);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "student_name" not in st.session_state:
    st.session_state.student_name = "Alex"
if "study_hours" not in st.session_state:
    st.session_state.study_hours = 4.0
if "attendance" not in st.session_state:
    st.session_state.attendance = 78.0

st.title("🎓 Student Success Predictor")
st.caption("Estimate whether a student is likely to pass based on study habits and attendance.")

with st.sidebar:
    st.header("Student profile")
    st.session_state.student_name = st.text_input("Student name", value=st.session_state.student_name)

    preset = st.radio("Quick profile", ["Custom", "Improving", "Balanced", "Strong"])
    if preset == "Improving":
        st.session_state.study_hours = 3.0
        st.session_state.attendance = 68.0
    elif preset == "Balanced":
        st.session_state.study_hours = 5.0
        st.session_state.attendance = 80.0
    elif preset == "Strong":
        st.session_state.study_hours = 7.0
        st.session_state.attendance = 92.0

    st.session_state.study_hours = st.slider(
        "Study hours per day",
        min_value=0.0,
        max_value=12.0,
        value=float(st.session_state.study_hours),
        step=0.5,
    )
    st.session_state.attendance = st.slider(
        "Attendance percentage",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.attendance),
        step=1.0,
    )

    st.markdown("---")
    st.caption("Current input")
    st.write(f"Name: {st.session_state.student_name}")
    st.write(f"Study hours: {float(st.session_state.study_hours):.1f} h/day")
    st.write(f"Attendance: {float(st.session_state.attendance):.0f}%")

name = st.session_state.student_name.strip() or "Student"
study_hours = float(st.session_state.study_hours)
attendance = float(st.session_state.attendance)

prediction, probability = calculate_prediction(study_hours, attendance)
status = "Pass" if prediction == 1 else "Fail"

with st.container():
    tab_overview, tab_analysis, tab_summary = st.tabs(["Overview", "Scenario view", "Insights"])

    with tab_overview:
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", status)
        col2.metric("Chance of passing", f"{probability * 100:.1f}%")
        col3.metric("Attendance", f"{attendance:.0f}%")

        if prediction == 1:
            st.success(f"✅ {name} is predicted to PASS with a {probability * 100:.1f}% probability.")
        else:
            st.error(f"❌ {name} is predicted to FAIL with a {probability * 100:.1f}% probability.")

        st.write(
            f"This forecast combines study hours and attendance to estimate how likely {name} is to succeed. "
            f"A prediction above 50% indicates a pass outcome."
        )

        st.progress(probability)

    with tab_analysis:
        chart_df = get_probability_curve(attendance)
        st.subheader("Pass probability by study hours")
        st.line_chart(chart_df.set_index("Study hours"), y="Pass probability")

        scenario_data = {
            "Improving": (3.0, 68.0),
            "Balanced": (5.0, 80.0),
            "Strong": (7.0, 92.0),
        }

        compare_rows = []
        for label, (hours, attendance_value) in scenario_data.items():
            scenario_prediction, scenario_probability = calculate_prediction(hours, attendance_value)
            compare_rows.append(
                {
                    "Scenario": label,
                    "Study hours": hours,
                    "Attendance": attendance_value,
                    "Prediction": "Pass" if scenario_prediction == 1 else "Fail",
                    "Chance": f"{scenario_probability * 100:.1f}%",
                }
            )

        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    with tab_summary:
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Student name",
                    "Study hours/day",
                    "Attendance",
                    "Predicted result",
                    "Pass probability",
                ],
                "Value": [
                    name,
                    f"{study_hours:.1f} hours",
                    f"{attendance:.0f}%",
                    status,
                    f"{probability * 100:.1f}%",
                ],
            }
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        st.markdown(
            """
            <div style='padding: 1rem; border-radius: 0.8rem; background: #eff6ff; border: 1px solid #dbeafe;'>
                <strong>Recommendation:</strong>
                Focus on consistent daily study time and maintaining attendance above 80% to keep the pass probability high.
            </div>
            """,
            unsafe_allow_html=True,
        )
