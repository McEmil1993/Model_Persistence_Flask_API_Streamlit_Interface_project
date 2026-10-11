"""Exercise 3: Streamlit interface for the student pass/fail model."""
import joblib
import pandas as pd
import streamlit as st
import altair as alt

DEFAULT_ATTENDANCE = 85  # model needs attendance; the form only asks for study hours


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

st.title("Student Pass/Fail Predictor")

name = st.text_input("Student name")
study_hours = st.number_input(
    "Study hours per week", min_value=0.0, max_value=40.0, value=5.0, step=0.5
)

if st.button("Predict"):
    features = pd.DataFrame(
        {"study_hours": [study_hours], "attendance_percentage": [DEFAULT_ATTENDANCE]}
    )
    result = int(model.predict(features)[0])
    prob_pass = float(model.predict_proba(features)[0][1])
    label = "Pass" if result == 1 else "Fail"
    who = name.strip() or "The student"

    # Result message
    if result == 1:
        st.success(f"{who} is predicted to **{label}** (pass probability {prob_pass:.0%}).")
    else:
        st.error(f"{who} is predicted to **{label}** (pass probability {prob_pass:.0%}).")

    # Explanation
    st.subheader("How was this prediction generated?")
    st.write(
        "A logistic regression model was trained earlier and saved to "
        "`student_pass_model.joblib`. This app loads that file and applies it "
        f"to the study hours you entered ({study_hours:g}), using a default "
        f"attendance of {DEFAULT_ATTENDANCE}%. The model outputs a pass "
        "probability, and 50% or higher is predicted as Pass. "
        "No retraining happens here."
    )

    # Bonus chart: study hours vs. predicted class
    hours_range = [h / 2 for h in range(0, 25)]  # 0 to 12 hours
    curve = pd.DataFrame(
        {"study_hours": hours_range, "attendance_percentage": DEFAULT_ATTENDANCE}
    )
    curve["pass_probability"] = model.predict_proba(curve)[:, 1]
    point = pd.DataFrame(
        {"study_hours": [study_hours], "pass_probability": [prob_pass], "predicted": [label]}
    )

    line = alt.Chart(curve).mark_line().encode(
        x=alt.X("study_hours", title="Study hours"),
        y=alt.Y("pass_probability", title="Pass probability", scale=alt.Scale(domain=[0, 1])),
    )
    dot = alt.Chart(point).mark_point(size=200, filled=True).encode(
        x="study_hours",
        y="pass_probability",
        color=alt.Color("predicted", legend=alt.Legend(title="Predicted class")),
    )
    st.subheader("Study hours vs. predicted class")
    st.altair_chart(line + dot)
