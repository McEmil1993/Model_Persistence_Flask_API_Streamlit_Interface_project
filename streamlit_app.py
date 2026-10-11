
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓"
)

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

st.title("Student Performance Prediction")
st.write("Predict a student's result based on study hours.")

student_name = st.text_input("Student Name")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=1.0
)

if st.button("Predict Result"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        input_data = pd.DataFrame({
            "study_hours": [study_hours]
        })

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours:g}")

        if prediction == "Pass":
            st.success("Predicted Result: PASS")
            bar_color = "green"
        else:
            st.error("Predicted Result: FAIL")
            bar_color = "red"

        st.write(
            "Explanation: The trained Decision Tree model "
            "uses study hours to predict the student's result. "
            "The prediction is based on patterns learned from "
            "the training dataset and is not a guarantee of "
            "actual academic performance."
        )

        st.subheader("Study Hours vs. Predicted Class")

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            [prediction],
            [study_hours],
            color=bar_color,
            width=0.5
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Study Hours")
        ax.set_title("Study Hours and Predicted Result")
        ax.set_ylim(0, max(10, study_hours + 2))

        ax.text(
            0,
            study_hours,
            f"{study_hours:g} hours",
            ha="center",
            va="bottom"
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
