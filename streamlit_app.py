from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Prediction", page_icon="🎓")
st.title("Student Pass Prediction")
st.write("Enter a student's name and study hours to estimate the result.")

if not MODEL_PATH.exists():
    st.error("Model file not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

student_name = st.text_input("Student name")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
)

if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = str(model.predict([[study_hours]])[0])
        st.subheader(f"Result for {student_name.strip()}: {prediction}")
        st.write(
            f"The model received {study_hours:g} study hours and used its "
            "learned relationship between study hours and the example pass/fail labels."
        )
        st.caption(
            "This is a classroom demonstration using sample data, not a reliable "
            "assessment of a real student's performance."
        )


st.divider()
st.subheader("Bonus: Study hours vs. predicted class")

import altair as alt

# Include every whole hour AND your exact input
hours_values = sorted(
    set([float(h) for h in range(25)] + [float(study_hours)])
)

# Predict the result for every hour
predictions = [
    str(model.predict([[hours]])[0])
    for hours in hours_values
]

chart_data = pd.DataFrame({
    "Study hours": hours_values,
    "Predicted class": [
        1 if p.strip().lower() == "pass" else 0
        for p in predictions
    ]
})

# Predict the exact current input
current_prediction = str(model.predict([[study_hours]])[0])
current_class = (
    1 if current_prediction.strip().lower() == "pass" else 0
)

# Data for the current input marker
input_data = pd.DataFrame({
    "Study hours": [float(study_hours)],
    "Predicted class": [current_class]
})

# Blue line: model predictions
line = alt.Chart(chart_data).mark_line(
    color="blue",
    strokeWidth=2
).encode(
    x=alt.X(
        "Study hours:Q",
        title="Study hours (0–24)",
        scale=alt.Scale(domain=[0, 24])
    ),
    y=alt.Y(
        "Predicted class:Q",
        title="Predicted class (0 = Fail, 1 = Pass)",
        scale=alt.Scale(domain=[-0.05, 1.05]),
        axis=alt.Axis(
            values=[0, 1],
            labelExpr="datum.value == 1 ? 'Pass (1)' : 'Fail (0)'"
        )
    ),
    tooltip=[
        alt.Tooltip("Study hours:Q", format=".2f"),
        alt.Tooltip("Predicted class:Q")
    ]
)

# Red dot: your exact input and its prediction
point = alt.Chart(input_data).mark_point(
    color="red",
    size=150,
    filled=True
).encode(
    x="Study hours:Q",
    y="Predicted class:Q",
    tooltip=[
        alt.Tooltip("Study hours:Q", format=".2f"),
        alt.Tooltip(
            "Predicted class:Q",
            title="Predicted class"
        )
    ]
)

# Combine the line and marker into one graph
chart = (line + point).properties(
    height=400
).interactive()

st.altair_chart(chart, use_container_width=True)

st.caption(
    f"Your input: {study_hours:g} study hours — "
    f"Predicted result: {current_prediction}."
)