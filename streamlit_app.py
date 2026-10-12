import joblib
import pandas as pd
import streamlit as st

st.title("Student Pass Predictor")

model = joblib.load("student_pass_model.joblib")

name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = int(model.predict([[hours]])[0])
        if pred == 1:
            st.success(f"{name} is predicted to PASS with {hours} study hours.")
        else:
            st.error(f"{name} is predicted to FAIL with {hours} study hours.")

        st.info(
            "The app loads a pre-trained model and applies it to the study hours you "
            "entered. The model learned from past examples that more study hours "
            "usually mean a higher chance of passing, so no retraining happens here."
        )

        # Bonus: chart comparing study hours with predicted class
        hours_range = [h / 2 for h in range(0, 21)]   # 0 to 10 hours
        preds = model.predict([[h] for h in hours_range])
        chart_df = pd.DataFrame({"Study hours": hours_range, "Predicted class": preds})
        st.subheader("Study hours vs predicted class (1 = Pass)")
        st.line_chart(chart_df, x="Study hours", y="Predicted class")