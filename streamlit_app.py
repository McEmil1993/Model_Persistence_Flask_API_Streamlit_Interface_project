"""Streamlit interface for the saved student pass/fail model."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="centered",
)

# Simple and readable UI styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
    }

    .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        color: #1e3a5f;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        color: #243b53;
    }

    p, label, .stMarkdown {
        color: #334155;
    }

    div[data-testid="stForm"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 10px;
    }

    .stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 7px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
        border: none;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 Student Pass/Fail Predictor")
st.caption("Student performance estimation")
st.write(
    "Enter the student's information below to estimate "
    "their likelihood of passing."
)

st.divider()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "The saved model was not found. Run `python train_model.py` "
        "in this folder first."
    )
    st.stop()

model = load_model()

# Student information section
st.subheader("Student Information")

name = st.text_input(
    "Student name",
    placeholder="Enter student name",
)

hours = st.number_input(
    "Study hours per day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
    help="Enter the number of hours spent studying per day.",
)

attendance = st.slider(
    "Attendance percentage",
    min_value=0,
    max_value=100,
    value=75,
    help="Select the student's attendance percentage.",
)

st.caption(f"Selected attendance: {attendance}%")

st.write("")

if st.button("Predict Student Result", width="stretch"):
    features = pd.DataFrame(
        [
            {
                "study_hours": hours,
                "attendance_percentage": attendance,
            }
        ]
    )

    prediction = int(model.predict(features)[0])
    pass_class_index = list(model.classes_).index(1)
    pass_probability = float(
        model.predict_proba(features)[0][pass_class_index]
    )

    result = "Pass" if prediction == 1 else "Fail"
    display_name = name.strip() or "Student"

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Student", display_name)

    with col2:
        st.metric("Estimated pass probability", f"{pass_probability:.1%}")

    if prediction == 1:
        st.success(f"Prediction: {result}")
    else:
        st.warning(f"Prediction: {result}")

    # Explanation section
    with st.expander("How was this prediction generated?", expanded=True):
        st.write(
            "The app uses the saved Logistic Regression model to evaluate "
            "the study hours and attendance entered above. It returns a "
            "predicted class and an estimated probability of passing. "
            "The model is not retrained for each prediction."
        )

    # Chart section
    st.divider()
    st.subheader("Study Hours and Pass Probability")

    st.write(
        "This chart shows how the estimated probability of passing changes "
        "with study hours while keeping attendance at the selected value."
    )

    hours_range = np.linspace(0, 24, 121)

    chart_features = pd.DataFrame(
        {
            "study_hours": hours_range,
            "attendance_percentage": [attendance] * len(hours_range),
        }
    )

    pass_probabilities = model.predict_proba(chart_features)[
        :, pass_class_index
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        hours_range,
        pass_probabilities,
        color="#2563eb",
        linewidth=2.5,
        label="Estimated pass probability",
    )

    ax.axhline(
        0.5,
        color="#94a3b8",
        linestyle="--",
        linewidth=1.2,
        label="50% reference line",
    )

    ax.scatter(
        [hours],
        [pass_probability],
        color="#dc2626",
        s=65,
        zorder=3,
        label="This student",
    )

    ax.set(
        xlim=(0, 24),
        ylim=(0, 1),
        xlabel="Study hours per day",
        ylabel="Estimated probability of passing",
        title=f"Attendance: {attendance}%",
    )

    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels(
        [f"{value:.0%}" for value in np.linspace(0, 1, 6)]
    )

    ax.grid(axis="both", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.divider()
st.caption("Student Pass/Fail Predictor | Model-based estimate")
