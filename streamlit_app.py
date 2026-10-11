"""Exercise 3 - Streamlit interface (study hours + attendance)."""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
model = joblib.load("student_pass_model_v2.joblib")   # trained on hours + attendance

st.title("🎓 Student Pass Predictor")

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=16.0,
                        value=3.0, step=0.5)
attendance = st.number_input("Attendance percentage (0-100)", min_value=0.0,
                             max_value=100.0, value=75.0, step=1.0)

if st.button("Predict"):
    # Validation
    if not name.strip():
        st.warning("Please enter the student's name.")
    elif hours < 0:
        st.error("Study hours cannot be negative.")
    elif not 0 <= attendance <= 100:
        st.error("Attendance percentage must be between 0 and 100.")
    else:
        X = pd.DataFrame({"study_hours": [hours],
                          "attendance_percentage": [attendance]})
        prediction = int(model.predict(X)[0])
        prob = model.predict_proba(X)[0][1]

        if prediction == 1:
            st.success(f"✅ {name} is predicted to PASS.")
        else:
            st.error(f"❌ {name} is predicted to FAIL.")

        st.write(
            f"**How this was generated:** the model is a logistic regression "
            f"trained on past students' study hours, attendance and results. "
            f"For {hours} hours of study and {attendance:.0f}% attendance it "
            f"estimates a {prob:.0%} chance of passing; above 50% means Pass."
        )

        # Bonus chart: pass probability across study hours at this attendance
        hours_range = [h / 2 for h in range(0, 33)]
        chart_X = pd.DataFrame({
            "study_hours": hours_range,
            "attendance_percentage": [attendance] * len(hours_range),
        })
        probs = model.predict_proba(chart_X)[:, 1]
        chart_df = pd.DataFrame({"Pass probability": probs}, index=hours_range)
        chart_df.index.name = "Study hours"
        st.subheader(f"Study hours vs. pass probability (at {attendance:.0f}% attendance)")
        st.line_chart(chart_df)
        st.caption(f"{name} studied {hours} h with {attendance:.0f}% attendance "
                   f"→ predicted class: {'Pass' if prediction else 'Fail'}")