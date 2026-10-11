"""Exercise 3 - Streamlit Interface (Rusel)

A simple interface so a non-technical user can use the model.

Run:  streamlit run streamlit_app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Use a local model file if present, otherwise the one built in exercise1
MODEL_FILE = Path(__file__).resolve().parent / "rusel_pass_model.joblib"
if not MODEL_FILE.exists():
    MODEL_FILE = (
        Path(__file__).resolve().parent.parent / "exercise1" / "rusel_pass_model.joblib"
    )

model = joblib.load(MODEL_FILE)

st.set_page_config(page_title="Student Pass / Fail Checker", page_icon=":books:")
st.title("Student Pass / Fail Checker")

name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, step=0.5)

if st.button("Predict"):
    if name.strip() == "":
        st.warning("Please enter the student's name.")
    else:
        row = pd.DataFrame({"study_hours": [hours]})
        label = "Pass" if model.predict(row)[0] == 1 else "Fail"

        st.success(f"{name} is predicted to {label}.")
        st.caption(
            "How this works: the app loads a saved model file and applies it "
            "to the study hours you entered. Nothing is trained again."
        )
        # Bonus: chart comparing the entered study hours with the predicted class
        st.bar_chart({"Your study hours": hours, "Example Pass level": 5.0})
