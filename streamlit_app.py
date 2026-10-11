# streamlit_app.py
# Exercise 3 - Streamlit Interface (with Bonus chart)
import os

import altair as alt
import joblib
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(HERE, "student_pass_model.joblib"))


def build_chart(model, entered_hours):
    """Bonus: compare study hours with the predicted class."""
    hours = [h / 2 for h in range(0, 25)]  # 0 to 12 in steps of 0.5
    classes = model.predict(pd.DataFrame(hours).values)
    df = pd.DataFrame({
        "Study hours": hours,
        "Predicted class": ["Pass" if c == 1 else "Fail" for c in classes],
    })
    # the bar closest to the entered hours is shown in full color, others are faded
    closest = min(hours, key=lambda h: abs(h - entered_hours))
    df["Highlight"] = df["Study hours"].apply(lambda h: 1.0 if h == closest else 0.35)

    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Study hours:O", title="Study hours"),
        y=alt.Y("Predicted class:N", sort=["Fail", "Pass"], title="Predicted class"),
        color=alt.Color("Predicted class:N",
                        scale=alt.Scale(domain=["Fail", "Pass"], range=["#e45756", "#54a24b"])),
        opacity=alt.Opacity("Highlight:Q", legend=None),
        tooltip=["Study hours", "Predicted class"],
    )


model = load_model()

st.set_page_config(page_title="Student Pass Predictor", page_icon="📘")
st.title("📘 Student Pass Predictor")
st.caption("Enter the student's name and study hours, then press Predict.")

name = st.text_input("Student name", placeholder="e.g. Juan Dela Cruz")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=4.0, step=0.5)

if st.button("Predict", type="primary"):
    if len(name.strip()) == 0:
        st.warning("Please enter the student's name.")
    else:
        chance = float(model.predict_proba([[hours]])[0][1])
        will_pass = chance >= 0.5

        # Result message
        if will_pass:
            st.success(f"{name} is likely to PASS.")
        else:
            st.error(f"{name} is likely to FAIL.")

        # Short explanation
        st.info(
            f"Explanation: the model learned from past examples of study hours and results. "
            f"With {hours} study hour(s), it estimates a {chance * 100:.0f}% chance of passing. "
            f"A chance of 50% or more is predicted as Pass; anything lower is Fail."
        )

        # Bonus chart
        st.subheader("Study hours vs predicted class")
        st.altair_chart(build_chart(model, hours), width="stretch")
        st.caption(f"Each bar is the model's prediction for that many study hours. "
                   f"The brightest bar is {name}'s input ({hours} hour(s)).")