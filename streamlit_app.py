import streamlit as st
import joblib
import pandas as pd

# Load model once
model = joblib.load("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Prediction App")
st.write("Enter the student's details below to predict whether they will pass.")

# --- Inputs ---
student_name = st.text_input("Student Name", placeholder="e.g., Juan Dela Cruz")
study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
attendance = st.slider("Attendance Percentage", min_value=0, max_value=100, value=75)

# --- Prediction ---
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = model.predict([[study_hours, attendance]])[0]
        label = "Pass" if pred == 1 else "Fail"
        color = "green" if pred == 1 else "red"

        st.markdown(f"### Result for *{student_name}*")
        st.markdown(f"<h2 style='color:{color};'>{label}</h2>", unsafe_allow_html=True)

        # Explanation
        st.info(
            f"The model used a Logistic Regression classifier trained on "
            f"study hours and attendance data. With *{study_hours} study hours* "
            f"and *{attendance}% attendance*, it predicted *{label}*."
        )

        # --- Bonus: Chart ---
        st.subheader("📊 Study Hours vs Predicted Class")
        chart_data = pd.DataFrame({
            "Study Hours": [study_hours],
            "Predicted Class (1=Pass, 0=Fail)": [int(pred)]
        })
        st.bar_chart(chart_data.set_index("Study Hours"))