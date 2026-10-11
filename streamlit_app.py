
from pathlib import Path
import html

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"
CSS_PATH = BASE_DIR / "static" / "style.css"


# ============================================================
# LOAD CSS
# ============================================================

if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")
else:
    st.warning(
        "CSS file not found. Check that static/style.css exists."
    )


# ============================================================
# LOAD SAVED MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_pass_model.joblib was not found. "
            "Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the saved model: {error}")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# HERO BANNER
# ============================================================

st.html("""
<div class="hero">
    <span class="hero-badge">
        MACHINE LEARNING DEMONSTRATION
    </span>

    <h1>🎓 Student Performance Predictor</h1>

    <p>
        Explore how a trained machine learning model generates
        student-result predictions based on study hours.
        Enter the student details below to get started.
    </p>
</div>
""")


# ============================================================
# STUDENT INFORMATION
# ============================================================

st.html("""
<div class="section-heading">
    <span class="section-number">01</span>
    <span>Student Information</span>
</div>

<p class="section-description">
    Enter the student's details to generate a prediction.
</p>
""")


with st.container(border=True):

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student's full name",
        key="student_name_input",
    )

    study_hours = st.number_input(
        "Daily Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5,
        help="Enter the number of hours the student studies daily.",
        key="study_hours_input",
    )

    st.caption(
        "The model uses study hours as its prediction input."
    )

    predict_clicked = st.button(
        "🎯 Predict Student Result",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# GENERATE PREDICTION
# ============================================================

if predict_clicked:

    if not student_name.strip():
        st.session_state.prediction_result = None
        st.warning("Please enter the student's name.")

    else:
        try:
            # The current model uses study hours only.
            predicted_class = model.predict(
                np.array([[study_hours]], dtype=float)
            )[0]

            st.session_state.prediction_result = {
                "student": student_name.strip(),
                "hours": float(study_hours),
                "prediction": str(predicted_class).strip(),
            }

        except Exception as error:
            st.session_state.prediction_result = None
            st.error(f"Prediction failed: {error}")


# ============================================================
# DISPLAY PREDICTION RESULT
# ============================================================

result = st.session_state.prediction_result

if result is not None:

    st.divider()

    st.html("""
    <div class="section-heading">
        <span class="section-number">02</span>
        <span>Prediction Result</span>
    </div>
    """)

    student = html.escape(result["student"])
    hours = result["hours"]
    prediction = result["prediction"]
    safe_prediction = html.escape(prediction)

    if prediction.lower() == "pass":
        result_class = "result-pass"
        result_icon = "✓"
    elif prediction.lower() == "fail":
        result_class = "result-fail"
        result_icon = "✕"
    else:
        result_class = "result-pass"
        result_icon = "ℹ"

    # Render the HTML card.
    st.html(f"""
    <div class="result-card">
        <div class="result-label">
            PREDICTED STUDENT RESULT
        </div>

        <div class="{result_class}">
            {result_icon} {safe_prediction}
        </div>

        <div class="result-detail">
            <p>
                Student: <strong>{student}</strong>
            </p>

            <p>
                Study hours:
                <strong>{hours:g} hours per day</strong>
            </p>
        </div>
    </div>
    """)

    # Render the explanation separately.
    st.html(f"""
    <div class="explanation">
        <strong>How was this prediction generated?</strong>

        <p>
            The application loaded the saved model from
            <code>student_pass_model.joblib</code> and used
            {hours:g} study hours to generate the prediction.
            The model was not retrained.
        </p>
    </div>
    """)


# ============================================================
# PREDICTION ANALYTICS
# ============================================================

st.divider()

st.html("""
<div class="section-heading">
    <span class="section-number">03</span>
    <span>Prediction Analytics</span>
</div>

<p><strong>Study Hours vs. Predicted Result</strong></p>

<p class="section-description">
    The line shows the model's predictions across different
    study hours. The gold star marks the selected study hours
    after a prediction is generated.
</p>
""")


try:
    hours_range = np.arange(0.0, 24.5, 0.5)

    chart_predictions = model.predict(
        hours_range.reshape(-1, 1)
    )

    def class_to_number(label):
        label = str(label).strip().lower()

        if label == "fail":
            return 0

        if label == "pass":
            return 1

        return 0.5

    prediction_values = [
        class_to_number(value)
        for value in chart_predictions
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.step(
        hours_range,
        prediction_values,
        where="mid",
        linewidth=2.5,
        label="Model prediction",
    )

    ax.set_title(
        "Prediction by Study Hours",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Daily Study Hours")
    ax.set_ylabel("Predicted Result")

    ax.set_xlim(0, 24)
    ax.set_ylim(-0.15, 1.15)

    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])

    ax.grid(True, linestyle="--", alpha=0.25)

    if result is not None:

        current_hours = result["hours"]
        current_value = class_to_number(
            result["prediction"]
        )

        ax.scatter(
            [current_hours],
            [current_value],
            marker="*",
            s=220,
            label="Selected study hours",
            zorder=5,
        )

        ax.axvline(
            current_hours,
            linestyle=":",
            alpha=0.6,
        )

    ax.legend(loc="best")
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

except Exception as error:
    st.warning(f"Unable to display the chart: {error}")


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
    <p>
        Student Performance Predictor ·
        Machine Learning Demonstration
    </p>

    <p>
        Powered by Python, Scikit-learn, and Streamlit
    </p>
</div>
""")
