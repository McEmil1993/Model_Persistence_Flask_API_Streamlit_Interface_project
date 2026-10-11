import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load saved model
model = joblib.load("student_pass_model.joblib")

# Header
st.title("🎓 Student Performance Prediction")
st.caption("Machine Learning | Model Persistence | Streamlit")

st.write(
    "Enter the student's information below to predict "
    "the result based on study hours."
)

st.divider()

# Student inputs
student_name = st.text_input(
    "Student Name",
    placeholder="Enter student name"
)

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=8.0,
    value=5.0,
    step=0.5
)

# Prediction button
if st.button("🔍 Predict Student Result", use_container_width=True):

    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        prediction = int(model.predict([[study_hours]])[0])

        probabilities = model.predict_proba([[study_hours]])[0]
        classes = list(model.classes_)

        # Find probability of PASS (class 1)
        pass_index = classes.index(1)
        pass_probability = probabilities[pass_index] * 100

        result = "PASS" if prediction == 1 else "FAIL"

        st.divider()
        st.subheader("📊 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Study Hours",
                f"{study_hours:g} hours"
            )

        with col2:
            st.metric(
                "Estimated Passing Probability",
                f"{pass_probability:.1f}%"
            )

        st.write(f"**Student Name:** {student_name.strip()}")

        if prediction == 1:
            st.success(f"🎉 Predicted Result: {result}")
        else:
            st.error(f"⚠️ Predicted Result: {result}")

        st.progress(int(round(pass_probability)))

        st.caption(
            "The percentage represents the model's estimated "
            "probability of PASS, not the student's actual grade "
            "or a guarantee of passing."
        )

        st.subheader("💡 Prediction Explanation")

        st.write(
            f"The model received {study_hours:g} study hours "
            f"as input and predicted **{result}**. "
            f"The estimated probability of PASS is "
            f"{pass_probability:.1f}%. "
            "This prediction uses the previously trained "
            "Decision Tree model."
        )

        st.subheader("📈 Study Hours Overview")

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            ["Study Hours"],
            [study_hours]
        )

        ax.set_ylabel("Hours")
        ax.set_title("Student's Study Hours")
        ax.set_ylim(0, 8)

        st.pyplot(fig)

        st.info(
            "Note: This model uses study hours only. "
            "Attendance percentage does not affect this prediction."
        )

st.divider()
st.caption("Student Pass Prediction System | Python + Scikit-learn")