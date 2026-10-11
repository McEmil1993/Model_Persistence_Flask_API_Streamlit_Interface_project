"""Exercise 3 — Streamlit Interface.

A small web form around the trained model: enter a student name and study
hours, get a prediction and an explanation of how it was produced.

Run:
    streamlit run streamlit_app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILENAME = "student_pass_model.joblib"

# Anchor to this script's folder so the app finds the model regardless of the
# directory Streamlit was launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    """Load the model once and cache it across reruns.

    Streamlit re-runs this whole script on every interaction, so without the
    cache the model would be read from disk on each keystroke.
    """
    return joblib.load(MODEL_PATH)


st.title("🎓 Student Pass Predictor")
st.write("Predict whether a student will pass based on their study hours.")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file `{MODEL_FILENAME}` not found. "
        "Run `python train_model.py` first to create it."
    )
    st.stop()

model = load_model()

# --- Inputs ------------------------------------------------------------------

student_name = st.text_input("Student name", value="")
study_hours = st.number_input(
    "Study hours per day", min_value=0.0, max_value=8.0, value=3.0, step=0.5
)
predict_clicked = st.button("Predict", type="primary")

# --- Prediction --------------------------------------------------------------

if predict_clicked:
    row = pd.DataFrame({"study_hours": [study_hours]})
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    label = "Pass" if prediction == 1 else "Fail"

    name = student_name.strip() or "This student"
    if prediction == 1:
        st.success(f"{name} is predicted to **{label}**.")
    else:
        st.warning(f"{name} is predicted to **{label}**.")

    # --- How the prediction was generated -----------------------------------

    st.subheader("How this prediction was generated")

    coefficient = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    weighted_sum = coefficient * study_hours + intercept

    st.write(
        "The model was trained on study hours alone. It learned a single "
        "coefficient and an intercept, and predicts by combining them with "
        "your input:"
    )
    st.latex(
        r"z = (\text{coefficient} \times \text{study hours}) + \text{intercept}"
    )
    st.write(
        f"With a coefficient of `{coefficient:.4f}` and an intercept of "
        f"`{intercept:.4f}`, your input of `{study_hours:.1f}` study hours per day gives:"
    )
    st.latex(rf"z = ({coefficient:.4f} \times {study_hours:.1f}) + {intercept:.4f} = {weighted_sum:.4f}")

    st.write(
        "That value is passed through the logistic function, which squashes it "
        "into a probability between 0 and 1. If the probability is at or above "
        "0.5 the model predicts **Pass**, otherwise **Fail**."
    )
    st.write(
        f"Probability of passing: **{probability:.1%}** — "
        f"{'at or above' if probability >= 0.5 else 'below'} the 0.5 threshold, "
        f"so the prediction is **{label}**."
    )

    # --- Bonus: study hours vs predicted class -------------------------------

    st.subheader("Study hours vs. predicted class")

    hour_grid = np.linspace(0, 8, 101)
    grid = pd.DataFrame({"study_hours": hour_grid})
    probabilities = model.predict_proba(grid)[:, 1]

    curve = pd.DataFrame(
        {"study_hours": hour_grid, "probability of passing": probabilities}
    )

    # Draw the probability curve, the 0.5 decision threshold, and the user's
    # input as a vertical marker. Vega-Lite is used (rather than st.line_chart)
    # because it supports these extra layers.
    st.vega_lite_chart(
        {
            "layer": [
                {
                    "data": {"values": curve.to_dict("records")},
                    "mark": {"type": "line", "color": "#1f77b4"},
                    "encoding": {
                        "x": {
                            "field": "study_hours",
                            "type": "quantitative",
                            "title": "Daily study hours",
                        },
                        "y": {
                            "field": "probability of passing",
                            "type": "quantitative",
                            "title": "Probability of passing",
                            "scale": {"domain": [0, 1]},
                        },
                    },
                },
                {
                    "data": {
                        "values": [
                            {
                                "study_hours": study_hours,
                                "label": f"{study_hours:.1f} hours",
                            }
                        ]
                    },
                    "mark": {"type": "rule", "color": "#d62728", "strokeDash": [6, 4]},
                    "encoding": {
                        "x": {"field": "study_hours", "type": "quantitative"},
                        "tooltip": {"field": "label"},
                    },
                },
                {
                    "data": {
                        "values": [
                            {"hours": study_hours, "probability": probability}
                        ]
                    },
                    "mark": {
                        "type": "point",
                        "color": "#d62728",
                        "filled": True,
                        "size": 100,
                    },
                    "encoding": {
                        "x": {"field": "hours", "type": "quantitative"},
                        "y": {"field": "probability", "type": "quantitative"},
                    },
                },
                {
                    "data": {"values": [{"threshold": 0.5}]},
                    "mark": {"type": "rule", "color": "#999999", "strokeDash": [4, 4]},
                    "encoding": {"y": {"field": "threshold", "type": "quantitative"}},
                },
            ],
        },
        width="stretch",
    )

    st.caption(
        f"The red dashed line marks your input of {study_hours:.1f} hours and the "
        "red dot shows its predicted probability."
    )
    st.caption(
        "The curve crosses 0.5 near 3 daily study hours — that is the model's "
        "decision boundary, where a prediction flips from Fail to Pass."
    )
