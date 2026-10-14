import streamlit as st
import joblib
import pandas as pd

model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Predictor")

name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=3.0)

if st.button("Predict"):
    prediction = int(model.predict([[hours]])[0])
    who = name if name else "The student"

    if prediction == 1:
        st.success(f"{who} is predicted to PASS with {hours} study hours.")
    else:
        st.error(f"{who} is predicted to FAIL with {hours} study hours.")

    st.write(
        "How this was generated: a logistic regression model was trained on "
        "example study hours and pass/fail results. It learned a pattern from "
        "that data and uses your study hours to predict the class (1 = pass, 0 = fail)."
    )

    # Bonus: chart of study hours vs predicted class
    hours_range = [h / 2 for h in range(0, 25)]
    predicted = [int(model.predict([[h]])[0]) for h in hours_range]
    chart_data = pd.DataFrame(
        {"Predicted class (1 = pass)": predicted}, index=hours_range
    )
    chart_data.index.name = "Study hours"
    st.subheader("Study hours vs predicted class")
    st.line_chart(chart_data)