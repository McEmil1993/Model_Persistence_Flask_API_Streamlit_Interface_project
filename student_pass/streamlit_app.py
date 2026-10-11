"""Exercise 3 - Streamlit interface."""
import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

st.title("🎓 Student Pass Predictor")
st.write("Enter a student's name and weekly study hours to predict pass or fail.")

# Inputs
name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

# Prediction button
if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        features = pd.DataFrame({"study_hours": [hours]})
        pred = int(model.predict(features)[0])
        prob = float(model.predict_proba(features)[0][1])

        # Result message
        if pred == 1:
            st.success(f"✅ {name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"❌ {name} is predicted to FAIL (pass probability {prob:.0%}).")

        # Short explanation
        st.subheader("How was this generated?")
        st.write(
            f"The app loaded a logistic regression model that was trained earlier on "
            f"past study-hours data and saved with joblib. It took the {hours} study hours "
            f"you entered and calculated a **{prob:.0%}** chance of passing. "
            f"Since that is {'50% or higher' if pred == 1 else 'below 50%'}, the prediction is "
            f"**{'Pass' if pred == 1 else 'Fail'}**. The model is not retrained here."
        )

        # Chart: study hours vs. predicted class (Pass / Fail)
        st.subheader("Study hours vs. predicted class")
        hours_range = pd.DataFrame({"study_hours": [x / 2 for x in range(0, 49)]})
        hours_range["pass_percent"] = model.predict_proba(hours_range[["study_hours"]])[:, 1] * 100
        hours_range["predicted_class"] = [
            "Pass" if p == 1 else "Fail" for p in model.predict(hours_range[["study_hours"]])
        ]

        x_enc = alt.X("study_hours:Q", title="Study hours")
        y_enc = alt.Y(
            "pass_percent:Q",
            title="Predicted result (%)",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(labelExpr="datum.value + '%'"),
        )
        class_color = alt.Color(
            "predicted_class:N",
            title="Predicted class",
            scale=alt.Scale(domain=["Fail", "Pass"], range=["#d62728", "#2ca02c"]),
        )

        # Every study-hours value, colored by the class the model predicts for it
        line = alt.Chart(hours_range).mark_line(color="lightgray").encode(x=x_enc, y=y_enc)
        points = (
            alt.Chart(hours_range)
            .mark_circle(size=60)
            .encode(
                x=x_enc, y=y_enc, color=class_color,
                tooltip=["study_hours", alt.Tooltip("pass_percent:Q", format=".1f"), "predicted_class"],
            )
        )

        # Dashed 50% line = the Fail/Pass cutoff
        cutoff = (
            alt.Chart(pd.DataFrame({"y": [50]}))
            .mark_rule(strokeDash=[5, 5], color="gray")
            .encode(y="y:Q")
        )

        # This student: black ring with their name and result
        student_df = pd.DataFrame({
            "study_hours": [hours],
            "pass_percent": [prob * 100],
            "label": [f"{name}: {'Pass' if pred == 1 else 'Fail'} ({prob:.0%})"],
        })
        ring = (
            alt.Chart(student_df)
            .mark_point(size=250, color="black", strokeWidth=3, filled=False)
            .encode(x=x_enc, y=y_enc)
        )
        label = (
            alt.Chart(student_df)
            .mark_text(align="left", dx=14, dy=-14, color="black", fontWeight="bold")
            .encode(x=x_enc, y=y_enc, text="label:N")
        )

        st.altair_chart(line + points + cutoff + ring + label, use_container_width=True)
        st.caption(
            "Green dots are study-hour values the model predicts as Pass, red dots as Fail. "
            f"The circled point is {name} ({hours} hours). Above the dashed 50% line = Pass."
        )