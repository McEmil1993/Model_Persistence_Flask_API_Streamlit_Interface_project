
import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Prediction", page_icon="🎓")

st.title("🎓 Student Pass Prediction")
st.write("Predict student performance based on study hours.")

# Student inputs
student_name = st.text_input("Student Name")

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=4.0,
    step=0.5
)

# Make a prediction
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        X = pd.DataFrame({"study_hours": [study_hours]})
        prediction = model.predict(X)[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")

        if str(prediction) == "1":
            st.success("Prediction: Pass")
        else:
            st.error("Prediction: Fail")

        st.info("This prediction is based on study hours only.")

# Bonus: Prediction Chart
st.divider()
st.subheader("📊 Bonus: Study Hours Prediction Chart")
st.write("Compare the model's predictions for 1 to 8 study hours.")

hours = list(range(1, 9))
chart_data = pd.DataFrame({
    "study_hours": hours
})

chart_data["prediction"] = model.predict(
    pd.DataFrame({"study_hours": hours})
)

chart_data["Result"] = chart_data["prediction"].map({
    0: "Fail",
    1: "Pass"
})

st.line_chart(
    chart_data,
    x="study_hours",
    y="prediction"
)

st.caption("Chart values: 0 = Fail, 1 = Pass, assuming these are your model's labels.")
st.dataframe(chart_data, hide_index=True)
