"""
Exercise 3 - Streamlit Interface

Features
    - Student name input
    - Study-hours input
    - Prediction button
    - Result message
    - Short explanation of how the prediction was generated
    - Bonus: chart comparing study hours with the predicted class

This app uses the study-hours-only model (student_pass_model.joblib), because
the interface only asks for study hours.

Run:  python train_model.py     (once, to create the model files)
      streamlit run streamlit_app.py
"""

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"
CHART_MAX_HOURS = 12


@st.cache_resource
def load_model():
    """Load the saved model once and reuse it (no retraining)."""
    return joblib.load(MODEL_PATH)


def predict(model, hours: float):
    """Return (predicted_class, pass_probability) for the given study hours."""
    row = pd.DataFrame({"study_hours": [hours]})
    predicted_class = int(model.predict(row)[0])
    pass_probability = float(model.predict_proba(row)[0][1])
    return predicted_class, pass_probability


def build_explanation(model, name: str, hours: float, probability: float) -> str:
    """Plain-language description of how the prediction was produced."""
    coefficient = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    score = intercept + coefficient * hours

    text = (
        f"The model is a **logistic regression** that was trained earlier on past "
        f"student data and loaded from `{MODEL_PATH.name}` (it is not retrained "
        f"when you click the button).\n\n"
        f"1. It combines the study hours into a score: "
        f"`{intercept:.2f} + {coefficient:.2f} x {hours:g} = {score:.2f}`.\n"
        f"2. It converts that score into a pass probability between 0 and 1: "
        f"**{probability:.0%}** for {name}.\n"
        f"3. If the probability is 50% or higher, the prediction is **Pass**; "
        f"otherwise it is **Fail**."
    )
    if coefficient > 0:
        threshold = -intercept / coefficient
        text += (
            f"\n\nFrom its training data, the model learned that about "
            f"**{threshold:.1f} study hours** is where a student reaches a 50% "
            f"chance of passing."
        )
    return text


def build_chart(model, hours: float, predicted_class: int) -> Figure:
    """Bonus chart: predicted class for a range of study hours, with the student marked."""
    sweep = np.linspace(0, max(CHART_MAX_HOURS, hours + 1), 200)
    classes = model.predict(pd.DataFrame({"study_hours": sweep}))

    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    ax.step(sweep, classes, where="post", color="tab:blue", label="Model prediction")
    ax.scatter(
        [hours],
        [predicted_class],
        s=140,
        color="tab:green" if predicted_class == 1 else "tab:red",
        zorder=3,
        label="This student",
    )
    ax.set_xlabel("Study hours")
    ax.set_ylabel("Predicted class")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail (0)", "Pass (1)"])
    ax.set_ylim(-0.2, 1.2)
    ax.set_title("Study hours vs predicted class")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
    st.title("🎓 Student Pass Predictor")
    st.write("Enter a student's details to predict whether they will pass.")

    if not MODEL_PATH.exists():
        st.error(
            f"Model file `{MODEL_PATH.name}` not found. "
            "Run `python train_model.py` first, then reload this page."
        )
        return

    model = load_model()

    # --- Inputs ---------------------------------------------------------------
    student_name = st.text_input("Student name")
    study_hours = st.number_input(
        "Study hours", min_value=0.0, max_value=24.0, value=4.0, step=0.5
    )

    # --- Prediction button ----------------------------------------------------
    if st.button("Predict"):
        name = student_name.strip()
        if not name:
            st.warning("Please enter the student's name.")
            return
        if not math.isfinite(study_hours):
            st.warning("Please enter a valid number of study hours.")
            return

        predicted_class, probability = predict(model, study_hours)

        # --- Result message ---------------------------------------------------
        if predicted_class == 1:
            st.success(
                f"✅ {name} is predicted to PASS with {study_hours:g} study hours "
                f"(pass probability {probability:.0%})."
            )
        else:
            st.error(
                f"❌ {name} is predicted to FAIL with {study_hours:g} study hours "
                f"(pass probability {probability:.0%})."
            )

        # --- Explanation ------------------------------------------------------
        st.subheader("How was this prediction made?")
        st.markdown(build_explanation(model, name, study_hours, probability))

        # --- Bonus chart ------------------------------------------------------
        st.subheader("Study hours vs predicted class")
        st.pyplot(build_chart(model, study_hours, predicted_class))


if __name__ == "__main__":
    main()