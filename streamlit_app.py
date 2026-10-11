import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load saved model
MODEL_FILE = "student_pass_model.joblib"
model = joblib.load(MODEL_FILE)

# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("Student Performance Prediction")
st.write("Enter the student's information and click Predict.")

# Student name
student_name = st.text_input(
    "Student Name",
    placeholder="Enter student name"
)

# Study hours
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.01,
    format="%.2f"
)

# Predict button
if st.button("Predict", type="primary"):

    # Make prediction
    prediction = model.predict([[study_hours]])[0]

    # =========================
    # PREDICTION RESULT
    # =========================

    st.subheader("Prediction Result")

    if prediction == "Pass":

        st.success(
            f"{student_name or 'Student'} is predicted to PASS."
        )

    else:

        st.error(
            f"{student_name or 'Student'} is predicted to FAIL."
        )

    # =========================
    # EXPLANATION
    # =========================

    st.subheader("Why was this result predicted?")

    if study_hours < 5:

        explanation = (
            f"The student entered {study_hours:.2f} study hours. "
            "Based on the dataset, students who studied below "
            "5 hours were generally classified as Fail. "
            "Because the entered study hours are below the "
            "level where the dataset starts showing Pass results, "
            "the model predicted Fail."
        )

    elif study_hours >= 5:

        explanation = (
            f"The student entered {study_hours:.2f} study hours. "
            "Based on the dataset, students who studied around "
            "5 hours or more were generally classified as Pass. "
            "Because the entered study hours are within the range "
            "associated with Pass results in the dataset, "
            "the model predicted Pass."
        )

    st.info(explanation)

    # =========================
    # MODEL EXPLANATION
    # =========================

    st.write(
        "**Important:** The prediction is based only on "
        "Study Hours because the current dataset and trained "
        "model use Study Hours as the input feature."
    )

    # =========================
    # GRAPH
    # =========================

    st.subheader("Study Hours vs. Prediction")

    if prediction == "Pass":
        prediction_value = 1
    else:
        prediction_value = 0

    reference_hours = [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
    ]

    reference_values = []

    for hours in reference_hours:

        result = model.predict([[hours]])[0]

        if result == "Pass":
            reference_values.append(1)
        else:
            reference_values.append(0)

    # Create graph
    fig, ax = plt.subplots()

    # Model prediction line
    ax.plot(
        reference_hours,
        reference_values,
        marker="o",
        linestyle="-",
        label="Model Prediction"
    )

    # User input point
    ax.scatter(
        [study_hours],
        [prediction_value],
        s=180,
        marker="D",
        label="Your Input"
    )

    # Labels
    ax.set_xlabel("Study Hours")

    ax.set_ylabel(
        "Prediction (0 = Fail, 1 = Pass)"
    )

    ax.set_yticks([0, 1])

    ax.set_yticklabels([
        "Fail",
        "Pass"
    ])

    ax.set_title(
        "Student Study Hours Prediction"
    )

    ax.grid(True)

    ax.legend()

    # Show graph
    st.pyplot(fig)

else:

    st.caption(
        "The graph and explanation will appear after you click Predict."
    )