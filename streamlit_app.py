
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

name = st.text_input("Student Name")
hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Result"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = int(model.predict([[hours]])[0])
        result = "Pass" if prediction == 1 else "Fail"

        st.subheader(f"Student: {name}")
        st.write(f"Study Hours: {hours}")
        st.success(f"Predicted Result: {result}")

        st.write(
            "This prediction is based on the trained sample model "
            "and study hours only."
        )

        chart_data = pd.DataFrame({
            "Study Hours": [hours],
            "Predicted Class (0=Fail, 1=Pass)": [prediction]
        })

        st.bar_chart(
            chart_data.set_index("Study Hours")
        )
