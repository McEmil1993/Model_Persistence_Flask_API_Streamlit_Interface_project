import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction System",
    layout="centered"
)

# --------------------------------------------------
# 2. PROFESSIONAL CYAN AND GOLD THEME
# --------------------------------------------------
CYAN = "#00CFE8"
GOLD = "#F4C542"
BACKGROUND = "#0E1117"
CARD_BACKGROUND = "#171D27"
WHITE = "#FFFFFF"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND};
    }}

    h1, h2, h3 {{
        color: {CYAN} !important;
    }}

    /* Primary buttons (normal + form submit) */
    div.stButton > button,
    div.stFormSubmitButton > button {{
        background-color: {CYAN} !important;
        color: {BACKGROUND} !important;
        border: 2px solid {GOLD} !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        min-height: 44px;
        transition: all 0.2s ease-in-out;
    }}

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {{
        background-color: {GOLD} !important;
        color: {BACKGROUND} !important;
        border-color: {CYAN} !important;
    }}

    div.stButton > button:focus,
    div.stFormSubmitButton > button:focus,
    div.stButton > button:active,
    div.stFormSubmitButton > button:active {{
        background-color: {GOLD} !important;
        color: {BACKGROUND} !important;
        border-color: {CYAN} !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 207, 232, 0.35) !important;
        outline: none !important;
    }}

    div[data-testid="stMetric"] {{
        background-color: {CARD_BACKGROUND};
        border: 1px solid {GOLD};
        border-left: 4px solid {CYAN};
        border-radius: 8px;
        padding: 14px;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {CYAN} !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {WHITE} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# 3. LOAD THE TRAINED MODEL
# --------------------------------------------------
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "student_pass_model.joblib"
)

try:
    model = joblib.load(MODEL_PATH)

except FileNotFoundError:
    st.error(
        "The trained model file was not found. "
        "Please run train_model.py first."
    )
    st.stop()

except Exception:
    st.error(
        "The saved model could not be loaded. "
        "Please run train_model.py again to recreate it."
    )
    st.stop()

# --------------------------------------------------
# 4. PAGE HEADER
# --------------------------------------------------
st.title("Student Performance Prediction System")

st.write(
    "Predict a student's result based on daily study hours "
    "using a trained Decision Tree model."
)

st.caption(
    "Demonstration system | Predictions are not guarantees "
    "of actual academic performance."
)

st.divider()

# --------------------------------------------------
# 5. STUDENT INPUTS
# --------------------------------------------------
st.subheader("Student Information")

with st.form("prediction_form"):
    student_name = st.text_input(
        "Student Name",
        placeholder="Enter the student's name"
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5,
        help="Enter a value between 0 and 24 hours."
    )

    submitted = st.form_submit_button(
        "Predict Student Result",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# 6. GENERATE AND DISPLAY THE PREDICTION
# --------------------------------------------------
if submitted:

    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        try:
          
            sample = pd.DataFrame({
                "study_hours": [study_hours]
            })

       
            prediction = model.predict(sample)[0]

            if prediction not in ("Pass", "Fail"):
                st.error(
                    "The model returned an unexpected result. "
                    "Please check the training data and model."
                )
                st.stop()

            current_class = (
                1 if prediction == "Pass" else 0
            )

          
            st.divider()
            st.subheader("Prediction Result")

            st.write(f"**Student:** {student_name.strip()}")

            result_col, hours_col = st.columns(2)

            with result_col:
                st.metric(
                    label="Predicted Result",
                    value=prediction.upper()
                )

            with hours_col:
                st.metric(
                    label="Study Hours per Day",
                    value=f"{study_hours:g} hours"
                )

            if prediction == "Pass":
                st.success("The model predicts: PASS")
            else:
                st.warning("The model predicts: FAIL")

        
            st.subheader("Prediction Explanation")

            st.info(
                "The saved Decision Tree model uses patterns "
                "learned from the training dataset to predict "
                "a result based on study hours. Attendance is "
                "not used by this model."
            )

            # --------------------------------------------------
            # 7. BONUS CHART
            # --------------------------------------------------
            st.divider()
            st.subheader("Study Hours vs. Predicted Class")

          
            hours_range = list(range(1, 11))

            chart_data = pd.DataFrame({
                "study_hours": hours_range
            })

            chart_data["predicted_result"] = model.predict(
                chart_data[["study_hours"]]
            )

            chart_data["predicted_class"] = (
                chart_data["predicted_result"] == "Pass"
            ).astype(int)

        
            fig, ax = plt.subplots(figsize=(9, 4.5))

            fig.patch.set_facecolor(BACKGROUND)
            ax.set_facecolor(CARD_BACKGROUND)

            ax.step(
                chart_data["study_hours"],
                chart_data["predicted_class"],
                where="mid",
                color=CYAN,
                linewidth=2.5,
                label="Model prediction pattern"
            )

         
            ax.scatter(
                [study_hours],
                [current_class],
                color=GOLD,
                edgecolors=WHITE,
                linewidths=1.5,
                s=150,
                zorder=5,
                label="Current student"
            )

            ax.set_title(
                "Study Hours vs. Predicted Class",
                color=CYAN,
                pad=14
            )

            ax.set_xlabel(
                "Study Hours per Day",
                color=WHITE
            )

            ax.set_ylabel(
                "Predicted Class",
                color=WHITE
            )

            ax.set_yticks([0, 1])
            ax.set_yticklabels(
                ["Fail (0)", "Pass (1)"],
                color=WHITE
            )

           
            max_x = max(10, int(study_hours) + 1)
            ax.set_xlim(0, max_x)
            ax.set_ylim(-0.2, 1.2)
            ax.set_xticks(range(0, max_x + 1))
            ax.tick_params(colors=WHITE)

            ax.grid(
                True,
                color="#596273",
                alpha=0.35
            )

            for spine in ax.spines.values():
                spine.set_color("#596273")

            legend = ax.legend(
                facecolor=CARD_BACKGROUND,
                edgecolor=GOLD
            )

            for legend_text in legend.get_texts():
                legend_text.set_color(WHITE)

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            
            st.subheader("Current Student")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.metric(
                    "Study Hours Entered",
                    f"{study_hours:g} hours"
                )

            with summary_col2:
                st.metric(
                    "Predicted Class",
                    str(current_class)
                )

            st.caption(
                "The cyan line shows model predictions for "
                "1–10 study hours. The gold marker identifies "
                "the current student's input and predicted class. "
                "Class values: 0 = Fail; 1 = Pass."
            )

            st.warning(
                "This model uses a small demonstration dataset. "
                "Its predictions should not be treated as a "
                "reliable assessment of actual academic performance."
            )

        except Exception:
            st.error(
                "The prediction could not be completed. "
                "Check that the saved model is compatible with "
                "your training script and dataset."
            )