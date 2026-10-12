import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Header
st.title("🎓 Student Performance Prediction")
st.caption("Machine Learning | Model Persistence | Streamlit")

st.write(
    "Enter the student's name and daily study hours "
    "to predict the result."
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
        # Predict using the trained model
        prediction = int(model.predict([[study_hours]])[0])
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
                "Predicted Result",
                result
            )

        st.write(f"**Student Name:** {student_name.strip()}")

        if prediction == 1:
            st.success("🎉 Predicted Result: PASS")
        else:
            st.error("⚠️ Predicted Result: FAIL")

        # Prediction explanation
        st.subheader("💡 Prediction Explanation")

        st.write(
            f"The model received {study_hours:g} study hours "
            f"per day and predicted **{result}**."
        )

        st.caption(
            "Class 0 = FAIL; Class 1 = PASS. "
            "This is a prediction, not a guarantee of actual academic results."
        )

        # Bonus graph: Study Hours vs. Predicted Class
        st.subheader("📈 Study Hours vs. Predicted Class")

        hours_range = list(range(9))

        predicted_classes = [
            int(model.predict([[hours]])[0])
            for hours in hours_range
        ]

        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.plot(
            hours_range,
            predicted_classes,
            marker="o",
            linewidth=2,
            label="Predicted Result"
        )

        ax.scatter(
            [study_hours],
            [prediction],
            s=140,
            marker="*",
            label=f"Selected: {study_hours:g} hours",
            zorder=3
        )

        ax.set_title(
            "Student Result Based on Study Hours",
            fontsize=13,
            pad=18
        )

        ax.set_xlabel("Study Hours per Day", labelpad=10)
        ax.set_ylabel("Predicted Result", labelpad=10)

        ax.set_xticks(range(9))
        ax.set_yticks(
            [0, 1],
            labels=["FAIL", "PASS"]
        )

        ax.set_xlim(-0.3, 8.3)
        ax.set_ylim(-0.2, 1.2)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

        ax.legend(loc="best")
        fig.tight_layout(pad=2)

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "The prediction is determined by the student's study hours. "
            "Attendance percentage is checked separately to ensure that the value is within the valid range."
        )

st.divider()
st.caption("Student Pass Prediction System | Python + Scikit-learn")
