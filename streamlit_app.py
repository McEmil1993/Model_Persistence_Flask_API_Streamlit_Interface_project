import streamlit as st
import joblib
import pandas as pd

model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Predictor")

name = st.text_input("Student name")
hours = st.number_input("Study hours", 0.0, 24.0, 5.0)
att = st.number_input("Attendance %", 0.0, 100.0, 80.0)

if st.button("Predict"):
    pred = int(model.predict([[hours, att]])[0])
    who = name if name else "This student"

    if pred == 1:
        st.success(f"{who} is likely to PASS.")
    else:
        st.error(f"{who} is likely to FAIL.")

    st.write("How it works: the model compares your study hours and "
             "attendance with patterns it learned from past student data.")

    # Bonus chart
    st.bar_chart(pd.DataFrame({"Value": [hours, pred]},
                              index=["Study hours", "Predicted class (0=fail, 1=pass)"]))
