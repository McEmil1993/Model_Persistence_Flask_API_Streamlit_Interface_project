import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURES = ["study_hours", "attendance_percentage"]

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


try:
    model = load_model()
except FileNotFoundError:
    st.error("Hindi makita ang model. Patakbuhin muna ang `python train_model.py`.")
    st.stop()

st.title("🎓 Student Pass Predictor")

student_name = st.text_input("Student name")
study_hours = st.number_input("Study hours per day", 0.0, 24.0, 5.0, 0.5)
attendance = st.slider("Attendance percentage", 0, 100, 80)

if st.button("Predict", type="primary"):
    if not student_name.strip():
        st.warning("Ilagay muna ang pangalan ng student.")
        st.stop()

    row = pd.DataFrame([[study_hours, attendance]], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    if prediction == 1:
        st.success(f"✅ {student_name} ay predicted na **PASS** ({probability:.0%}).")
    else:
        st.error(f"❌ {student_name} ay predicted na **FAIL** ({probability:.0%} lang ang pass chance).")

    st.subheader("Paano nagawa ang prediction?")
    st.write(
        f"Nag-load ang app ng logistic regression model na na-train na at na-save gamit ang joblib. "
        f"Kinuha nito ang **{study_hours:g} study hours** at **{attendance}% attendance**, "
        f"kinalkula ang tsansang pumasa (**{probability:.0%}**), at nag-predict ng *pass* "
        f"kapag 50% pataas. Walang training na nangyayari dito."
    )

    st.subheader("Study hours vs. predicted class")
    hours = np.linspace(0, 10, 100)
    curve = pd.DataFrame({"study_hours": hours, "attendance_percentage": attendance})
    curve["pass_probability"] = model.predict_proba(curve[FEATURES])[:, 1]

    line = alt.Chart(curve).mark_line().encode(
        x=alt.X("study_hours", title="Study hours"),
        y=alt.Y("pass_probability", title="Probability of passing",
                scale=alt.Scale(domain=[0, 1])),
    )
    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        strokeDash=[5, 5], color="gray").encode(y="y")
    point = alt.Chart(pd.DataFrame({
        "study_hours": [study_hours],
        "pass_probability": [probability],
        "class": ["Pass" if prediction else "Fail"],
    })).mark_point(size=200, filled=True).encode(
        x="study_hours", y="pass_probability",
        color=alt.Color("class", scale=alt.Scale(
            domain=["Pass", "Fail"], range=["green", "red"])),
    )
    st.altair_chart(line + rule + point)