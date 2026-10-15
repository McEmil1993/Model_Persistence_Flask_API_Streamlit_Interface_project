
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Pass Prediction")

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
        prediction = model.predict([[hours]])[0]

        if prediction == "Pass":
            st.success(f"{name} is predicted to PASS!")
        else:
            st.error(f"{name} is predicted to FAIL.")

        st.write(f"Student Name: {name}")
        st.write(f"Study Hours: {hours}")
        st.write(f"Predicted Result: {prediction}")

        st.subheader("How was the prediction generated?")
        st.write(
            "The system uses a previously trained Logistic Regression "
            "model. It compares the entered study hours with patterns "
            "learned from the sample training data to predict Pass or Fail. "
            "This is a prediction, not a guarantee of actual performance."
        )

        st.subheader("Study Hours vs. Predicted Class")

        class_value = 1 if prediction == "Pass" else 0
        chart = pd.DataFrame({
            "Study Hours": [hours],
            "Predicted Class (1 = Pass, 0 = Fail)": [class_value]
        })

        st.bar_chart(
            chart,
            x="Study Hours",
            y="Predicted Class (1 = Pass, 0 = Fail)"
        )
