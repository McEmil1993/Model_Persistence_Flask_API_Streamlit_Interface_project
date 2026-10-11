import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Student Pass Predictor", page_icon="📝", layout="wide")

TEAL = "#0f766e"
AMBER = "#f59e0b"
RED = "#e11d48"
GREEN = "#16a34a"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #f0fdfa; }}
    [data-testid="stSidebar"] {{ background-color: #ccfbf1; }}
    .banner {{
        border-left: 8px solid {AMBER};
        background: white;
        padding: 18px 22px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }}
    .banner h1 {{ margin: 0; color: {TEAL}; font-size: 2.1rem; }}
    .banner p  {{ margin: 4px 0 0; color: #475569; }}
    .verdict {{
        padding: 22px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 14px;
    }}
    div.stButton > button {{
        width: 100%;
        background: {TEAL};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem;
        font-weight: 600;
    }}
    div.stButton > button:hover {{ background: #115e59; color: white; }}
    </style>
    <div class="banner">
        <h1>📝 Student Pass Predictor</h1>
        <p>A quick pass or fail estimate based on how a student studies.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Load the two saved models (no training happens here)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    simple = joblib.load("student_pass_model.joblib")
    full = joblib.load("student_pass_model_attendance.joblib")
    return simple, full


hours_only_model, hours_and_attendance_model = load_models()


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def read_attendance(raw_text):
    """Return (value, error). Blank text means attendance was not given."""
    raw_text = raw_text.strip()
    if raw_text == "":
        return None, None
    try:
        value = float(raw_text)
    except ValueError:
        return None, "Attendance must be a number."
    if value < 0 or value > 100:
        return None, "Attendance must be between 0 and 100."
    return value, None


def run_prediction(study_hours, attendance):
    """Pick the right model, return (label, probability, description)."""
    if attendance is None:
        features = pd.DataFrame({"study_hours": [study_hours]})
        chosen = hours_only_model
        description = "study hours only"
    else:
        features = pd.DataFrame(
            {"study_hours": [study_hours], "attendance_percentage": [attendance]}
        )
        chosen = hours_and_attendance_model
        description = "study hours and attendance"

    label = int(chosen.predict(features)[0])
    chance = float(chosen.predict_proba(features)[0][1])
    return chosen, label, chance, description


def draw_class_chart(chosen, study_hours, attendance, chance, label, student):
    """Show which study hours lead to FAIL or PASS, and where this student sits."""
    hours_range = np.linspace(0, 12, 121)
    if attendance is None:
        table = pd.DataFrame({"study_hours": hours_range})
        subtitle = "Predicted class for each study-hours value"
    else:
        table = pd.DataFrame(
            {
                "study_hours": hours_range,
                "attendance_percentage": np.full(len(hours_range), attendance),
            }
        )
        subtitle = f"Predicted class by study hours (attendance fixed at {attendance:g}%)"

    classes = chosen.predict(table)

    fig, ax = plt.subplots(figsize=(7, 3.2))

    # Coloured zones: red where the model says FAIL, green where it says PASS
    ax.fill_between(hours_range, 0, 1, where=(classes == 0),
                    color=RED, alpha=0.18, step="mid", label="FAIL zone")
    ax.fill_between(hours_range, 0, 1, where=(classes == 1),
                    color=GREEN, alpha=0.18, step="mid", label="PASS zone")

    # This student's study hours
    marker_color = GREEN if label == 1 else RED
    ax.axvline(study_hours, color=marker_color, linewidth=3)
    ax.text(
        study_hours, 0.5,
        f" {student}\n {study_hours:g} h",
        color=marker_color, fontsize=10, fontweight="bold", va="center",
        ha="left" if study_hours < 9 else "right",
    )

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Study hours")
    ax.set_title(subtitle)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Sidebar: all inputs live here
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Student details")
    student_name = st.text_input("Name")
    study_hours = st.number_input(
        "Study hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5
    )
    attendance_raw = st.text_input(
        "Attendance % (optional)", value="", placeholder="Leave blank to skip"
    )
    go = st.button("Predict")
    st.caption("Attendance is optional. If left blank, only study hours are used.")


# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
if not go:
    st.info("Fill in the details on the left, then press **Predict**.")
else:
    attendance_value, problem = read_attendance(attendance_raw)
    if not student_name.strip():
        problem = "Please enter the student's name."

    if problem:
        st.warning(problem)
    else:
        model_used, label, chance, used_text = run_prediction(study_hours, attendance_value)

        # Big verdict banner
        verdict_color = GREEN if label == 1 else RED
        verdict_text = "PASS" if label == 1 else "FAIL"
        st.markdown(
            f'<div class="verdict" style="background:{verdict_color};">'
            f"{student_name} is predicted to {verdict_text} ({chance:.0%})</div>",
            unsafe_allow_html=True,
        )

        tab_summary, tab_chart, tab_how = st.tabs(["Summary", "Chart", "How it works"])

        with tab_summary:
            c1, c2, c3 = st.columns(3)
            c1.metric("Study hours", f"{study_hours:g}")
            c2.metric(
                "Attendance",
                f"{attendance_value:g}%" if attendance_value is not None else "Not given",
            )
            c3.metric("Chance of passing", f"{chance:.0%}")
            st.progress(chance)

        with tab_chart:
            figure = draw_class_chart(
                model_used, study_hours, attendance_value, chance, label, student_name
            )
            st.pyplot(figure)

        with tab_how:
            st.write(
                f"The prediction used **{used_text}**. "
                "The model learned from past students' results and gives a probability "
                "of passing. A probability of 50% or higher is predicted as PASS, "
                "anything lower is predicted as FAIL."
            )
            st.write(
                "The model was saved earlier with joblib, so it is simply loaded here "
                "and does not need to be trained again."
            )