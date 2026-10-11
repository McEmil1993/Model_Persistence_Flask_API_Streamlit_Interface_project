"""Student success dashboard for the sample pass prediction model."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Success Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        from train_model import train_and_save_model

        train_and_save_model()
    return joblib.load(MODEL_PATH)


def predict_student(study_hours: float, attendance: float) -> int:
    model = get_model()
    features = pd.DataFrame(
        {
            "study_hours": [study_hours],
            "attendance_percentage": [attendance],
        }
    )
    return int(model.predict(features)[0])


with st.sidebar:
    st.markdown("## 🎓 Student Success")
    st.caption("Learning support dashboard")
    st.divider()
    st.markdown("**How to use**")
    st.write("Enter a student's details to view a model estimate and study-hours trend.")
       


st.title("Student outlook")
st.caption("Review study habits and attendance to support a timely conversation.")

left, right = st.columns([1.05, 1.45], gap="large")

with left:
    st.subheader("Student details")
    with st.form("student_prediction_form"):
        student_name = st.text_input("Student name", placeholder="e.g. Alex Morgan")
        study_hours = st.slider(
            "Study hours per day",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="Average hours spent studying each day.",
        )
        attendance = st.slider(
            "Attendance",
            min_value=0,
            max_value=100,
            value=90,
            step=1,
            format="%d%%",
            help="Percentage of classes attended.",
        )
        submitted = st.form_submit_button("Generate outlook", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction overview")
    if submitted:
        prediction = predict_student(study_hours, float(attendance))
        label = "Likely to pass" if prediction == 1 else "May need support"
        display_name = student_name.strip() or "This student"

        if prediction == 1:
            st.success(f"### {label}\n\n{display_name}'s current inputs align with a passing prediction.", icon="✅")
        else:
            st.warning(f"### {label}\n\nConsider checking in with {display_name} about study support.", icon="⚠️")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Study time", f"{study_hours:g} hrs/day")
        metric2.metric("Attendance", f"{attendance}%")
        metric3.metric("Model result", "Pass" if prediction == 1 else "Fail")

        st.markdown("#### Study-hours trend")
        st.caption(f"Predicted class across study hours, holding attendance at {attendance}%.")
        x_values = np.linspace(0, 10, 101)
        chart_features = pd.DataFrame(
            {
                "study_hours": x_values,
                "attendance_percentage": float(attendance),
            }
        )
        y_values = get_model().predict(chart_features)

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.step(x_values, y_values, where="mid", color="#2563EB", linewidth=2.5)
        ax.scatter(
            [study_hours],
            [prediction],
            color="#F97316",
            edgecolor="white",
            linewidth=1.5,
            s=115,
            zorder=3,
            label="Selected student",
        )
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0, 1], ["May need support", "Likely to pass"])
        ax.set_xlabel("Study hours per day")
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("How this estimate was generated"):
            st.write(
                "The app loads the saved logistic regression model and supplies the "
                "entered study hours and attendance percentage. The model was trained "
                "on both features in the sample CSV. Its output is a simplified "
                "classroom example and should be considered alongside other information."
            )
    else:
        st.info("Complete the student details and select **Generate outlook** to see a prediction.", icon="📋")
        st.markdown("#### What the dashboard shows")
        st.write(
            "A pass/fail estimate, the values used by the model, and a chart showing "
            "how the predicted class changes with study hours at the selected attendance level."
        )

st.divider()
st.caption("Student Success Dashboard · Educational demo · Sample data is synthetic")
