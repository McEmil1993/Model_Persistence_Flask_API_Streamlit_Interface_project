"""Exercise 3 - Streamlit interface for the student pass predictor.

Takes a student name and study hours, predicts PASS/FAIL on button click, shows
a result message and an explanation of how the prediction was made, and draws a
chart comparing study hours with the predicted class (bonus).
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"


def load_model():
    """Load the persisted model. No retraining happens in the UI."""
    return joblib.load(MODEL_PATH)


def build_chart_data(model, max_hours: float = 8.0) -> pd.DataFrame:
    """Return study hours (0.5 steps) and the predicted class for each."""
    hours = [step / 2 for step in range(int(max_hours * 2) + 1)]
    return pd.DataFrame(
        {
            "Study hours": [f"{h:g}" for h in hours],
            "Predicted class": [
                int(c) for c in model.predict([[h] for h in hours])
            ],
        }
    )


def main() -> None:
    st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
    st.title("🎓 Student Pass Predictor")
    st.caption("Predict whether a student passes based on daily study hours.")

    model = load_model()

    # --- Inputs --------------------------------------------------------------
    name = st.text_input("Student name")
    hours = st.number_input(
        "Study hours per day", min_value=0.0, max_value=24.0, value=4.0, step=0.5
    )

    if not st.button("Predict", type="primary"):
        return

    if not name.strip():
        st.warning("Please enter the student's name.")
        return

    # --- Result message ------------------------------------------------------
    prediction = int(model.predict([[hours]])[0])
    if prediction == 1:
        st.success(
            f"{name} is predicted to **PASS** (studying {hours} hours per day)."
        )
    else:
        st.error(
            f"{name} is predicted to **FAIL** (studying {hours} hours per day)."
        )

    # --- Explanation ---------------------------------------------------------
    st.markdown(
        f"""
#### How this prediction was generated
1. You entered the student name **{name}** and **{hours}** study hours per day.
2. The previously trained model was loaded from `{MODEL_PATH.name}` - it was
   **not** retrained.
3. The study-hours value was fed to the model, which applied the formula it
   learned during training: `sigmoid(coefficient x hours + intercept)`.
4. The output was mapped to a class: **1 = PASS**, **0 = FAIL**.
"""
    )

    # --- Bonus: chart of study hours vs. predicted class ---------------------
    st.subheader("Bonus: study hours vs. predicted class")
    st.bar_chart(build_chart_data(model).set_index("Study hours")["Predicted class"])
    st.caption(
        "0 = FAIL, 1 = PASS. Bars show the predicted class for each amount "
        "of study hours."
    )


main()
