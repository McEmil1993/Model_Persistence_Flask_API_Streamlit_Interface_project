from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon=":material/model_training:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #dfe3e8;
            --surface: #ffffff;
            --canvas: #f7f8fa;
            --blue: #255fc7;
            --green: #17845c;
        }

        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }

        .block-container {
            max-width: 880px;
            padding-top: 3.25rem;
            padding-bottom: 2.5rem;
        }

        .app-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1.5rem;
            margin-bottom: 0.9rem;
            padding: 0.35rem 0 1rem;
            border-bottom: 1px solid var(--line);
        }

        .app-label {
            margin: 0 0 0.25rem;
            color: var(--blue);
            font-family: Consolas, "Courier New", monospace;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .app-title {
            margin: 0;
            color: var(--ink);
            font-size: clamp(1.8rem, 4vw, 2.45rem);
            font-weight: 720;
            letter-spacing: -0.035em;
            line-height: 1.1;
        }

        .app-copy {
            max-width: 600px;
            margin: 0.45rem 0 0;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .model-state {
            display: inline-flex;
            flex: 0 0 auto;
            align-items: center;
            gap: 0.45rem;
            margin-top: 0.2rem;
            color: var(--green);
            font-size: 0.82rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .model-state::before {
            content: "";
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 50%;
            background: var(--green);
        }

        .model-flow {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0 0 1rem;
            color: var(--muted);
            font-family: Consolas, "Courier New", monospace;
            font-size: 0.78rem;
        }

        .model-flow strong { color: var(--ink); }
        .model-flow span { color: #98a2b3; }

        [data-testid="stForm"] {
            padding: 1rem 1rem 0.9rem;
            border: 1px solid var(--line);
            border-radius: 10px;
            background: var(--surface);
            box-shadow: none;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
            min-height: 44px;
            border-radius: 7px;
        }

        div[data-testid="stFormSubmitButton"] button {
            min-height: 44px;
            border-radius: 7px;
            background: var(--blue);
            box-shadow: none;
            font-weight: 700;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: #1f50aa;
        }

        div[data-testid="stFormSubmitButton"] button:focus-visible {
            outline: 3px solid rgba(37, 95, 199, 0.3);
            outline-offset: 2px;
        }

        .result-heading {
            margin: 1.15rem 0 0.2rem;
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 720;
        }

        div[data-testid="stMetric"] {
            min-height: 96px;
            padding: 0.8rem 0.9rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface);
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.55rem;
        }

        [data-testid="stAlert"] { border-radius: 8px; }
        [data-testid="stTabs"] button { min-height: 42px; }

        .fine-print {
            margin-top: 0.8rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.5;
        }

        @media (max-width: 640px) {
            .block-container { padding: 3.25rem 0.9rem 2rem; }
            .app-header { display: block; }
            .model-state { margin-top: 0.7rem; }
            .model-flow { flex-wrap: wrap; }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                transition-duration: 0.01ms !important;
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="app-header">
        <div>
            <p class="app-label">IT Elective 3</p>
            <h1 class="app-title">Student outcome predictor</h1>
            <p class="app-copy">
                Load the saved model and predict a result from study hours.
            </p>
        </div>
        <div class="model-state">Model ready</div>
    </header>
    <div class="model-flow" aria-label="Prediction process">
        <strong>study hours</strong><span>→</span>
        <strong>saved model</strong><span>→</span>
        <strong>pass or fail</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("The saved model was not found. Run train_model.py first.")
    st.stop()

model = load_model()

with st.form("prediction_form"):
    name_column, hours_column = st.columns([1.6, 1], gap="medium")

    with name_column:
        student_name = st.text_input(
            "Student name",
            placeholder="Example: Alex Rivera",
            help="Used only to personalize the result.",
        )

    with hours_column:
        study_hours = st.number_input(
            "Study hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
            help="Enter a value from 0 to 24.",
        )

    predict_clicked = st.form_submit_button(
        "Generate prediction",
        type="primary",
        icon=":material/insights:",
        width="stretch",
    )

if predict_clicked:
    cleaned_name = student_name.strip()

    if not cleaned_name:
        st.toast(
            "Enter the student's name before generating a prediction.",
            icon=":material/error:",
            duration="short",
        )
    else:
        safe_name = escape(cleaned_name)
        input_data = [[study_hours]]
        predicted_class = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        class_probabilities = {
            int(model_class): float(probability)
            for model_class, probability in zip(model.classes_, probabilities)
        }
        pass_probability = class_probabilities.get(1, 0.0)
        fail_probability = class_probabilities.get(0, 0.0)
        confidence = max(pass_probability, fail_probability)
        prediction = "Pass" if predicted_class == 1 else "Fail"

        st.markdown(
            f'<h2 class="result-heading">Result for {safe_name}</h2>',
            unsafe_allow_html=True,
        )

        if prediction == "Pass":
            st.success(f"{cleaned_name} is predicted to pass.", icon=":material/check_circle:")
        else:
            st.warning(f"{cleaned_name} is predicted to fail.", icon=":material/warning:")

        metric_prediction, metric_hours, metric_confidence = st.columns(3)
        metric_prediction.metric("Outcome", prediction)
        metric_hours.metric("Study hours", f"{study_hours:g} h")
        metric_confidence.metric("Confidence", f"{confidence:.1%}")

        confidence_tab, activity_tab, explanation_tab = st.tabs(
            ["Confidence", "Activity chart", "Explanation"]
        )

        with confidence_tab:
            confidence_data = pd.DataFrame(
                {
                    "Outcome": ["Fail", "Pass"],
                    "Probability": [fail_probability, pass_probability],
                }
            )
            st.bar_chart(
                confidence_data,
                x="Outcome",
                y="Probability",
                color="Outcome",
                y_label="Probability",
            )
            st.dataframe(
                confidence_data.style.format({"Probability": "{:.1%}"}),
                hide_index=True,
                width="stretch",
            )

        with activity_tab:
            activity_data = pd.DataFrame(
                {
                    "Measure": ["Study hours", "Predicted class"],
                    "Value": [study_hours, predicted_class],
                }
            )
            st.bar_chart(
                activity_data,
                x="Measure",
                y="Value",
                color="Measure",
                y_label="Value",
            )
            st.caption("Predicted class: 0 = Fail, 1 = Pass.")

        with explanation_tab:
            st.markdown(
                """
                The application loaded `student_pass_model.joblib` and sent the
                study-hours value to the saved Logistic Regression model. The
                model returned **1 for Pass** or **0 for Fail** without retraining.
                """
            )

        st.markdown(
            """
            <p class="fine-print">
                Educational use only. Study hours are the model's only feature;
                this result cannot determine a student's actual performance.
            </p>
            """,
            unsafe_allow_html=True,
        )
