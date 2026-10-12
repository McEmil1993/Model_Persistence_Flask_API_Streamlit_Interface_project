
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction System")
st.write("Predict a student's result using study hours.")

name = st.text_input("Student Name")

hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

if st.button("Predict Result"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        X = pd.DataFrame({"study_hours": [hours]})
        prediction = int(model.predict(X)[0])

        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction Result")
        st.write(f"Student Name: {name}")
        st.write(f"Study Hours: {hours}")

        if prediction == 1:
            st.success(f"Prediction: {result}")
        else:
            st.error(f"Prediction: {result}")

        st.write(
            "The prediction uses a previously trained model "
            "and the study hours entered by the student."
        )

        chart = pd.DataFrame({
            "Category": ["Study Hours", "Predicted Class"],
            "Value": [hours, prediction]
        })

        fig, ax = plt.subplots()
        ax.bar(chart["Category"], chart["Value"])
        ax.set_title("Study Hours vs Predicted Class")
        ax.set_ylabel("Value")
        st.pyplot(fig)
        plt.close(fig)
