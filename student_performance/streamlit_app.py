"""Streamlit interface for student performance predictions."""

import streamlit as st

from .model import load_model, predict_outcome

MIN_STUDY_HOURS = 0.0
MAX_STUDY_HOURS = 11.0
CHART_STEPS_PER_HOUR = 4
DECISION_THRESHOLD_PERCENT = 50.0


@st.cache_resource
def get_model():
    return load_model()


def main() -> None:
    """Render the Streamlit prediction experience."""
    st.set_page_config(
        page_title="Student Performance",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.caption("APPLIED MACHINE LEARNING  /  EXERCISE")
    st.title("STUDENT PERFORMANCE")
    st.write("A focused outcome estimate based on daily study time.")

    try:
        model = get_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.form("prediction_form"):
        student_name = st.text_input("Student name", placeholder="Enter a name", max_chars=80)
        study_hours = st.number_input(
            "Daily study hours",
            min_value=MIN_STUDY_HOURS,
            max_value=MAX_STUDY_HOURS,
            value=5.0,
            step=0.5,
            help="The sample training data covers 0 to 11 study hours per day.",
        )
        submitted = st.form_submit_button("Generate estimate", type="primary")

    if submitted:
        if not student_name.strip():
            st.warning("Enter a student name to continue.")
        else:
            outcome = predict_outcome(model, study_hours)
            st.divider()
            st.subheader(f"{student_name.strip()} — {outcome}")
            st.write(
                "The saved model was loaded for this prediction. It uses study hours only; "
                "the name is display-only, and attendance does not affect the result."
            )
            st.caption("Demonstration only — the model is trained on synthetic sample data.")

            chart_hours = [
                step / CHART_STEPS_PER_HOUR
                for step in range(int(MAX_STUDY_HOURS * CHART_STEPS_PER_HOUR) + 1)
            ]
            pass_class_index = list(model.classes_).index("Pass")
            pass_probabilities = model.predict_proba(
                [[hours] for hours in chart_hours]
            )[:, pass_class_index]
            selected_probability = float(
                model.predict_proba([[study_hours]])[0][pass_class_index] * 100
            )
            chart_data = {
                "Study hours": chart_hours,
                "Pass probability": (pass_probabilities * 100).tolist(),
                "Decision threshold": [DECISION_THRESHOLD_PERCENT] * len(chart_hours),
            }
            st.subheader("PASS PROBABILITY BY STUDY TIME")
            st.line_chart(
                chart_data,
                x="Study hours",
                y=["Pass probability", "Decision threshold"],
                color=["#0A0A0A", "#A3A3A3"],
                x_label="Daily study hours",
                y_label="Probability of passing (%)",
                height=280,
                width="stretch",
            )
            st.caption(
                f"At {study_hours:g} hours: {selected_probability:.1f}% estimated pass probability. "
                "The gray line marks the 50% decision threshold."
            )
