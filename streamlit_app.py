import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered"
)

MODEL_PATH = "student_pass_model.joblib"


# ---------------------------------------------------------
# Load the model once and cache it
# ---------------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)


model = load_model(MODEL_PATH)

LABELS = {0: "Fail", 1: "Pass"}

# ---------------------------------------------------------
# Title and intro
# ---------------------------------------------------------
st.title("🎓 Student Pass Predictor")
st.write(
    "Enter the student's details below, then click **Predict** to see whether "
    "the student is likely to pass."
)

st.divider()

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
student_name = st.text_input("Student name", placeholder="e.g. Juan Dela Cruz")

study_hours = st.number_input(
    "Study hours per week",
    min_value=0.0,
    max_value=40.0,
    value=5.0,
    step=0.5,
    help="How many hours the student studies per week."
)

attendance = st.slider(
    "Attendance percentage",
    min_value=0,
    max_value=100,
    value=80,
    step=1,
    help="Percentage of classes the student attended."
)

st.caption(
    "Note: this model was trained on **two** features — study hours and attendance — "
    "so both inputs are required."
)

# ---------------------------------------------------------
# Prediction button
# ---------------------------------------------------------
predict_clicked = st.button("Predict", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Result
# ---------------------------------------------------------
if predict_clicked:
    if not student_name.strip():
        st.warning("Please enter the student's name before predicting.")
    else:
        # Feature order must match training: [study_hours, attendance_percentage]
        features = np.array([[study_hours, attendance]], dtype=float)

        prediction_code = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        prob_fail = float(probabilities[0])
        prob_pass = float(probabilities[1])
        label = LABELS[prediction_code]

        st.divider()

        # ---------- Result message ----------
        if prediction_code == 1:
            st.success(f"✅ **{student_name}** is predicted to **PASS**.")
        else:
            st.error(f"❌ **{student_name}** is predicted to **FAIL**.")

        # ---------- Key numbers ----------
        col1, col2, col3 = st.columns(3)
        col1.metric("Study hours", f"{study_hours:g} h")
        col2.metric("Attendance", f"{attendance}%")
        col3.metric("Probability of passing", f"{prob_pass * 100:.1f}%")

        # ---------- Short explanation ----------
        st.subheader("How this prediction was generated")

        comparator = "greater than or equal to" if prob_pass >= 0.5 else "less than"

        st.markdown(
            f"""
1. **Inputs collected** — study hours = `{study_hours:g}`, attendance = `{attendance}%`.
2. **Model loaded** — a **Logistic Regression** classifier loaded from `student_pass_model.joblib`.
   It was trained on 130 synthetic student records.
3. **Probability computed** — the model returned:
   - P(Fail) = **{prob_fail:.4f}**
   - P(Pass) = **{prob_pass:.4f}**
4. **Decision rule applied** — the model assigns the class with the higher probability,
   which is the same as saying: if P(Pass) is {comparator} 0.5, predict **Pass**.
5. **Result** — P(Pass) = {prob_pass:.4f}, so the predicted class is **{label}**.

The model learned from historical data that students who studied at least **5 hours per week**
**and** attended at least **75%** of classes tended to pass.
            """
        )

        # ---------- Bonus: chart ----------
        st.subheader("Study hours vs. predicted class")

        hours_range = np.linspace(0, 12, 49)
        grid = np.column_stack([
            hours_range,
            np.full_like(hours_range, attendance, dtype=float)
        ])
        pass_probabilities = model.predict_proba(grid)[:, 1]

        chart_df = pd.DataFrame({
            "Study hours": hours_range,
            "Probability of passing": pass_probabilities,
            "Decision threshold (0.5)": np.full_like(hours_range, 0.5),
        }).set_index("Study hours")

        st.line_chart(chart_df)

        st.caption(
            f"Attendance is held constant at **{attendance}%**. "
            f"The **blue line** is the model's probability of passing as study hours increase. "
            f"The **orange line** is the 0.5 decision threshold. "
            f"Your input of **{study_hours:g} hours** gives P(Pass) = **{prob_pass:.2f}**, "
            f"which places it **{'above' if prob_pass >= 0.5 else 'below'}** the threshold."
        )