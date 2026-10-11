import streamlit as st
import matplotlib.pyplot as plt
from load_model import load_saved_model, predict_student

st.set_page_config(
    page_title="Student Result Checker",
    page_icon="📊",
    layout="centered"
)

model = load_saved_model()

# Clean white interface
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
}
.block-container {
    max-width: 760px;
    padding-top: 2rem;
}
h1, h2, h3, h4, p, label,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"] {
    color: #243447 !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    color: #243447 !important;
    background-color: #ffffff !important;
}
div.stButton > button {
    background-color: #243b53;
    color: #ffffff !important;
    border-radius: 7px;
    width: 100%;
    height: 44px;
    border: none;
}
div.stButton > button:hover {
    background-color: #334e68;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Student Result Checker")
st.caption("Machine Learning | Model Persistence")
st.write("Enter the student's information to generate a prediction.")
st.divider()

with st.container(border=True):
    st.subheader("Student Information")

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter full name"
    )

    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=8.0,
        value=5.0,
        step=0.5
    )

    predict_button = st.button("Predict Result")

if predict_button:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        result = predict_student(model, study_hours)

        # Calculate an estimated percentage from study hours.
        # This is an estimate, not actual attendance or pass probability.
        estimated_percentage = (study_hours / 8) * 100
        estimated_percentage = max(
            0.0,
            min(100.0, estimated_percentage)
        )

        st.divider()
        st.subheader("Prediction Result")

        st.write(f"**Student:** {student_name.strip()}")
        st.write(f"**Study Hours:** {study_hours:g} hours")
        st.write(
            f"**Estimated Percentage:** {estimated_percentage:.1f}%"
        )

        if result == "PASS":
            st.success("Predicted Result: PASS")
        else:
            st.error("Predicted Result: FAIL")

        st.info(
            "The Decision Tree model uses study hours to predict "
            "PASS or FAIL. The estimated percentage is calculated "
            "from study hours and is not actual attendance or "
            "the probability of passing."
        )

        st.subheader("Study Hours vs. Predicted Class")

        predicted_class = 1 if result == "PASS" else 0

        figure, axis = plt.subplots(figsize=(7, 3.5))

        bars = axis.bar(
            ["Study Hours", "Predicted Class"],
            [study_hours, predicted_class],
            color=["#2e7d32", "#66bb6a"],
            width=0.55
        )

        axis.set_ylabel("Value", color="#243447")
        axis.set_title(
            "Study Hours vs. Predicted Class",
            color="#243447"
        )
        axis.set_ylim(0, 8)

        axis.tick_params(axis="x", colors="#243447")
        axis.tick_params(axis="y", colors="#243447")

        axis.bar_label(
            bars,
            fmt="%.1f",
            padding=3,
            color="#243447"
        )

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#d7dee7")
        axis.spines["bottom"].set_color("#d7dee7")
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.set_axisbelow(True)

        axis.text(
            0.5,
            -0.20,
            "Predicted Class: 0 = FAIL | 1 = PASS",
            ha="center",
            color="#243447",
            transform=axis.transAxes
        )

        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

st.divider()
st.caption("Caitom Student Prediction System")