import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the updated model
model = joblib.load("student_pass_model.joblib")

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")

st.write(
    "Predict student performance using study hours "
    "and attendance percentage."
)

# Student information
student_name = st.text_input("Student Name")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance_percentage = st.number_input(
    "Attendance Percentage (%)",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=1.0
)

# Predict button
if st.button("Predict Student Performance"):

    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        student_data = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance_percentage": attendance_percentage
        }])

        prediction = model.predict(student_data)[0]

        st.subheader("Prediction Result")
        st.write(f"**Student Name:** {student_name}")
        st.write(f"**Study Hours:** {study_hours}")
        st.write(
            f"**Attendance:** {attendance_percentage}%"
        )

        if prediction == "Pass":
            st.success("Prediction: PASS")
            st.write(
                "The model predicts that the student is likely to pass."
            )
        else:
            st.error("Prediction: FAIL")
            st.write(
                "The model predicts that the student is likely to fail."
            )

        st.caption(
            "This prediction uses both study hours and attendance. "
            "It is based on illustrative sample training data."
        )

# Bonus chart
st.divider()
st.subheader("📊 Bonus: Study Hours vs. Predicted Result")

st.write(
    "The chart uses the attendance percentage you select above."
)

hours_range = pd.DataFrame({
    "study_hours": [float(h) for h in range(0, 11)],
    "attendance_percentage": [
        float(attendance_percentage)
    ] * 11
})

chart_predictions = model.predict(hours_range)

chart_data = pd.DataFrame({
    "Study Hours": hours_range["study_hours"],
    "Predicted Result": [
        1 if result == "Pass" else 0
        for result in chart_predictions
    ]
})

fig, ax = plt.subplots()

ax.plot(
    chart_data["Study Hours"],
    chart_data["Predicted Result"],
    marker="o"
)

ax.set_xlabel("Study Hours")
ax.set_ylabel("Predicted Result")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Fail", "Pass"])
ax.set_title(
    f"Predictions at {attendance_percentage}% Attendance"
)
ax.grid(True)

st.pyplot(fig)
plt.close(fig)