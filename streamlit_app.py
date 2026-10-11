"""Exercise 3 - Streamlit Interface
Run with:  streamlit run streamlit_app.py
(Run train_model.py first so student_pass_model.joblib exists.)
"""
import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

st.title("🎓 Student Pass Predictor")
st.write("Enter a student's details to predict whether they will pass or fail.")

# ---- Inputs ----
name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

# ---- Prediction button ----
if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        features = pd.DataFrame({"study_hours": [hours]})
        pred = int(model.predict(features)[0])
        prob = float(model.predict_proba(features)[0][1])
        label = "Pass" if pred == 1 else "Fail"

        # ---- Result message ----
        if pred == 1:
            st.success(f"{name} is predicted to **{label}** (probability of passing: {prob:.0%}).")
        else:
            st.error(f"{name} is predicted to **{label}** (probability of passing: {prob:.0%}).")

        # ---- Short explanation ----
        st.subheader("How was this prediction generated?")
        st.write(
            f"The app loaded a saved Logistic Regression model that had already learned a "
            f"pattern from past student data: more study hours means a higher chance of passing. "
            f"It took **{hours} study hours**, calculated a pass probability of **{prob:.0%}**, "
            f"and predicted **{label}** because the probability is "
            f"{'at least' if pred == 1 else 'below'} 50%. "
            f"The model was not retrained; it only applied what it had learned."
        )

        # ---- Bonus chart: study hours vs predicted class ----
        st.subheader("Study hours vs. predicted class")
        grid = pd.DataFrame({"study_hours": [x / 2 for x in range(0, 25)]})
        grid["predicted_class"] = ["Pass" if p == 1 else "Fail" for p in model.predict(grid)]
        student = pd.DataFrame({"study_hours": [hours], "predicted_class": [label], "who": [name]})

        base = alt.Chart(grid).mark_circle(size=60, opacity=0.5).encode(
            x=alt.X("study_hours:Q", title="Study hours"),
            y=alt.Y("predicted_class:N", title="Predicted class"),
            color=alt.Color("predicted_class:N", legend=None),
        )
        point = alt.Chart(student).mark_point(size=250, filled=True, color="black").encode(
            x="study_hours:Q", y="predicted_class:N", tooltip=["who", "study_hours", "predicted_class"]
        )
        st.altair_chart(base + point, use_container_width=True)
        st.caption("Faded dots = model's predictions for other study hours. Black marker = this student.")
