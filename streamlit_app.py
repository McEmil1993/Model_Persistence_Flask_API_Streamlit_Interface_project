"""Exercise 3 - Streamlit interface."""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
model = joblib.load("student_pass_model.joblib")

st.title("🎓 Student Pass Predictor")

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=16.0,
                        value=3.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        X = pd.DataFrame({"study_hours": [hours]})
        prediction = int(model.predict(X)[0])
        prob = model.predict_proba(X)[0][1]

        if prediction == 1:
            st.success(f"✅ {name} is predicted to PASS.")
        else:
            st.error(f"❌ {name} is predicted to FAIL.")

        st.write(
            f"**How this was generated:** the model is a logistic regression "
            f"trained on past students' study hours and results. For {hours} hours "
            f"it estimates a {prob:.0%} chance of passing; above 50% means Pass."
        )

        # Bonus chart: pass probability across study hours, with this student marked
        hours_range = [h / 2 for h in range(0, 33)]
        probs = model.predict_proba(pd.DataFrame({"study_hours": hours_range}))[:, 1]
        chart_df = pd.DataFrame({"Pass probability": probs}, index=hours_range)
        chart_df.index.name = "Study hours"
        st.subheader("Study hours vs. pass probability")
        st.line_chart(chart_df)
        st.caption(f"{name} studied {hours} h → predicted class: "
                   f"{'Pass' if prediction else 'Fail'}")
