"""
streamlit_app.py

Exercise 3 - Streamlit Interface
Uses the study-hours model saved by train_model.py (no re-training).

Run:  streamlit run streamlit_app.py
"""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "student_pass_model.joblib"
DATA_FILE = BASE_DIR / "students.csv"
LABELS = {0: "Fail", 1: "Pass"}
MAX_HOURS = 12.0

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


# The file's modified time is part of the cache key, so if you re-train the model
# (a new .joblib file) the app picks it up without restarting Streamlit.
@st.cache_resource
def load_model(modified_time):
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_training_data(modified_time):
    return pd.read_csv(DATA_FILE)


missing = [f.name for f in (MODEL_FILE, DATA_FILE) if not f.exists()]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Run `python train_model.py` first.")
    st.stop()

model = load_model(MODEL_FILE.stat().st_mtime)
training_data = load_training_data(DATA_FILE.stat().st_mtime)
coef = float(model.coef_[0][0])        # learned weight for study hours
intercept = float(model.intercept_[0])  # learned starting value

st.title("🎓 Student Pass Predictor")
st.write("Enter a student's study hours to predict whether they will pass or fail.")

# --- Inputs (in a form, so pressing Enter or the button submits once) --------
with st.form("predict_form"):
    student_name = st.text_input("Student name", max_chars=50, placeholder="e.g. Juan Dela Cruz")
    study_hours = st.number_input(
        "Study hours", min_value=0.0, max_value=MAX_HOURS, value=4.0, step=0.5
    )
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    name = student_name.strip() or "The student"

    sample = pd.DataFrame({"study_hours": [study_hours]})
    predicted_class = int(model.predict(sample)[0])
    pass_probability = float(model.predict_proba(sample)[0][1])
    result = LABELS[predicted_class]

    # --- Result message -----------------------------------------------------
    message = (
        f"**{name}** studied for **{study_hours:g} hours**. "
        f"Prediction: **{result}** ({pass_probability:.0%} chance of passing)."
    )
    if predicted_class == 1:
        st.success(message)
    else:
        st.error(message)

    # --- Short explanation --------------------------------------------------
    st.subheader("How was this prediction generated?")
    st.write(
        "The app loaded a Logistic Regression model that was already trained and "
        f"saved as `{MODEL_FILE.name}`, so no training happens here. The model "
        "turns the study hours into a pass probability. If the probability is 50% "
        "or higher, the result is **Pass**; otherwise it is **Fail**. For this student:"
    )
    score = intercept + coef * study_hours
    st.code(
        f"score = {intercept:.2f} + {coef:.2f} x {study_hours:g} hours = {score:.2f}\n"
        f"pass probability = 1 / (1 + e^-score) = {pass_probability:.0%}",
        language=None,
    )

    # --- Bonus: chart comparing study hours with the predicted class --------
    st.subheader("Study hours vs. predicted class")

    hours_grid = pd.DataFrame({"study_hours": [i / 10 for i in range(0, int(MAX_HOURS * 10) + 1)]})
    hours_grid["pass_probability"] = model.predict_proba(hours_grid)[:, 1]

    x_axis = alt.X("study_hours:Q", title="Study hours", scale=alt.Scale(domain=[0, MAX_HOURS]))
    y_axis = alt.Y("pass_probability:Q", title="Pass probability", scale=alt.Scale(domain=[0, 1]))

    curve = alt.Chart(hours_grid).mark_line(color="gray").encode(x=x_axis, y=y_axis)

    threshold = (
        alt.Chart(pd.DataFrame({"pass_probability": [0.5]}))
        .mark_rule(strokeDash=[5, 5], color="lightgray")
        .encode(y="pass_probability:Q")
    )

    past_students = training_data.assign(
        pass_probability=training_data["passed"].astype(float),
        result=training_data["passed"].map(LABELS),
    )
    past_points = (
        alt.Chart(past_students)
        .mark_circle(size=50, opacity=0.45)
        .encode(
            x="study_hours:Q",
            y="pass_probability:Q",
            color=alt.Color(
                "result:N",
                title="Past students",
                scale=alt.Scale(domain=["Fail", "Pass"], range=["#d62728", "#2ca02c"]),
            ),
        )
    )

    this_student = pd.DataFrame(
        {"study_hours": [study_hours], "pass_probability": [pass_probability], "name": [name]}
    )
    student_point = (
        alt.Chart(this_student)
        .mark_point(shape="diamond", size=250, filled=True, color="#1f77b4", opacity=1)
        .encode(x="study_hours:Q", y="pass_probability:Q", tooltip=["name", "study_hours", "pass_probability"])
    )

    # Background zones: where the model predicts Fail (red) and Pass (green).
    boundary = -intercept / coef if coef > 0 else None
    show_zones = boundary is not None and 0 < boundary < MAX_HOURS
    layers = [threshold, curve, past_points, student_point]
    if show_zones:
        zones = pd.DataFrame(
            {"start": [0.0, boundary], "end": [boundary, MAX_HOURS], "zone": ["Fail", "Pass"]}
        )
        zone_layer = (
            alt.Chart(zones)
            .mark_rect(opacity=0.12)
            .encode(
                x=alt.X("start:Q", scale=alt.Scale(domain=[0, MAX_HOURS])),
                x2="end:Q",
                color=alt.Color(
                    "zone:N",
                    legend=None,
                    scale=alt.Scale(domain=["Fail", "Pass"], range=["#d62728", "#2ca02c"]),
                ),
            )
        )
        layers.insert(0, zone_layer)

    st.altair_chart(alt.layer(*layers))
    zone_note = f" (Pass starts at about {boundary:.1f} study hours)" if show_zones else ""
    st.caption(
        "The blue diamond is this student. "
        "Dots are past students (top = passed, bottom = failed). "
        "The curve is the model's pass probability for each number of study hours. "
        f"The red area is where the model predicts Fail and the green area is where it predicts Pass{zone_note}."
    )
