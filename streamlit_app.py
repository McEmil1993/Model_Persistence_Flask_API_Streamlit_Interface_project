"""Exercise 3: interactive student prediction page."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from model_utils import get_prediction

st.set_page_config(page_title="Student Pass Predictor", page_icon="📘")
st.title("📘 Student Pass Predictor")
st.write("Fill in the details below to view a sample prediction.")

with st.form("student_prediction_form"):
    name = st.text_input("Student name", placeholder="Enter student name")
    hours = st.number_input(
        "Study hours per day", min_value=0.0, max_value=24.0,
        value=4.0, step=0.5
    )
    attendance = st.slider(
        "Attendance percentage", min_value=0, max_value=100, value=80
    )
    submitted = st.form_submit_button("Predict result")

if submitted:
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        output = get_prediction(hours, attendance)
        is_pass = output["prediction"] == 1

        st.subheader(f"Result for {name.strip()}")
        if is_pass:
            st.success("Predicted class: Pass")
        else:
            st.warning("Predicted class: Needs improvement")

        st.write(f"Study hours: **{hours:g} hours/day** · Attendance: **{attendance}%**")
        st.write(f"Estimated pass probability: **{output['pass_probability_percent']}%**")
        st.info(
            "How this prediction was generated: the app loaded a previously trained "
            "Logistic Regression model and provided the study-hours and attendance "
            "values. The model compared these inputs with patterns learned from its "
            "example training data. It did not train again when Predict was clicked."
        )

        label = "Pass" if is_pass else "Needs improvement"
        chart = pd.DataFrame({
            "Measure": ["Study hours (per day)", "Predicted class (0/1)"],
            "Value": [hours, output["prediction"]],
        })
        st.caption(
            f"Bonus chart — predicted class: {label} "
            "(1 = Pass, 0 = Needs improvement)"
        )
        figure, axis = plt.subplots()
        axis.bar(chart["Measure"], chart["Value"])
        axis.set_ylabel("Value")
        axis.set_title("Study hours and predicted class")
        axis.tick_params(axis="x", labelrotation=10)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
