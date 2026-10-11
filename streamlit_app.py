"""Exercise 3 - Streamlit interface (uses the study-hours model)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"


def train_study_model():
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_study_model()


model = load_model()

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0,
                        value=5.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])

        if pred == 1:
            st.success(f"{name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"{name} is predicted to FAIL (pass probability {prob:.0%}).")

        st.info(
            "How this was generated: a logistic regression model, trained earlier "
            "on example study-hours data, was loaded from a .joblib file. It turns "
            f"{hours} study hours into a pass probability; above 50% it predicts "
            "PASS, otherwise FAIL. The model was not retrained for this prediction."
        )

        # Bonus: probability curve across study hours, with this student marked
        grid = np.linspace(0, 12, 50)
        curve = pd.DataFrame(
            {"Pass probability": model.predict_proba(grid.reshape(-1, 1))[:, 1]},
            index=pd.Index(grid.round(2), name="Study hours"),
        )
        st.subheader("Study hours vs. predicted class")
        st.line_chart(curve)
        st.caption(f"{name}: {hours} hours -> {'Pass (1)' if pred else 'Fail (0)'}")
