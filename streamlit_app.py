
"""Student Success Predictor - Streamlit Dashboard."""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

model = joblib.load("student_pass_attendance_model.joblib")

# ---------- DESIGN ----------
st.markdown("""
<style>
:root {
    color-scheme: light;
}

.stApp {
    background: #F6F7F1;
    color: #263F35;
}


[data-testid="stHeader"] {
    background: #F6F7F1;
    height: 3.5rem;
}

.block-container {
    max-width: 1180px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}


h1, h2, h3, p, label, li {
    color: #263F35;
}

h1 {
    font-size: 2.6rem !important;
    font-weight: 750 !important;
    letter-spacing: -1px;
}

h2, h3 {
    font-weight: 650 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E1E7DE;
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 4px 14px rgba(38, 63, 53, 0.045);
}

div[data-testid="stWidgetLabel"] p {
    color: #405B4C !important;
    font-weight: 600;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #263F35 !important;
    border: 1px solid #D5DED4 !important;
    border-radius: 9px !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: #78947B !important;
    box-shadow: 0 0 0 1px #78947B !important;
}

div[data-testid="stNumberInput"] button {
    background: #EDF2E9 !important;
    color: #365744 !important;
    border: none !important;
}

div.stButton > button {
    width: 100%;
    min-height: 48px;
    background: #6F8F75;
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    font-weight: 650;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background: #52745B;
    border: none;
    color: #FFFFFF !important;
}

div[data-testid="stMetric"] {
    background: #F0F4ED;
    border: 1px solid #E0E8DC;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #526D5B !important;
}

div[data-testid="stMetricValue"] {
    color: #263F35 !important;
    font-weight: 750;
}

hr {
    border-color: #E1E7DE;
}

.small-note {
    color: #718078;
    font-size: 0.88rem;
}

.eyebrow {
    color: #66836C;
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.result-pass {
    background: #EAF3E9;
    border: 1px solid #D5E5D2;
    border-radius: 14px;
    padding: 22px;
}

.result-fail {
    background: #FBEDEF;
    border: 1px solid #F1D5DA;
    border-radius: 14px;
    padding: 22px;
}

.result-pass h2, .result-fail h2 {
    margin: 0;
    font-size: 2rem;
}

.result-pass p, .result-fail p {
    margin-bottom: 0;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown(
    '<div class="eyebrow">STUDENT PERFORMANCE ANALYTICS</div>',
    unsafe_allow_html=True
)
st.title("Student Success Predictor")
st.markdown(
    "Explore how study habits and attendance relate to a student's "
    "predicted academic outcome."
)
st.markdown(
    '<p class="small-note">A machine learning demonstration · '
    'Study smart, attend consistently, keep growing.</p>',
    unsafe_allow_html=True
)
st.divider()

# ---------- INPUTS AND PREDICTION ----------
input_col, result_col = st.columns([1, 1.15], gap="large")

with input_col:
    with st.container(border=True):
        st.subheader("👤 Student information")
        name = st.text_input(
            "Student name",
            placeholder="Enter full name"
        )

        hours = st.number_input(
            "Study hours per day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
            help="Average number of hours spent studying per day."
        )

        attendance = st.number_input(
            "Attendance percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
            help="Percentage of classes attended."
        )

        predict_clicked = st.button(
            "Generate prediction  →",
            use_container_width=True
        )

with result_col:
    with st.container(border=True):
        st.subheader("📊 Prediction summary")

        if predict_clicked:
            if not name.strip():
                st.warning("Please enter the student's name.")
            else:
                # Feature order matches the training data:
                # [study_hours, attendance_percentage]
                student_data = [[hours, attendance]]
                prediction = model.predict(student_data)[0]
                probability = model.predict_proba(student_data)[0][1]
                label = "Pass" if prediction == 1 else "Fail"

                if prediction == 1:
                    st.markdown(
                        f'<div class="result-pass">'
                        f'<h2>✓ &nbsp; {label}</h2>'
                        f'<p>{name.strip()} is predicted to pass.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-fail">'
                        f'<h2>× &nbsp; {label}</h2>'
                        f'<p>{name.strip()} may need additional support.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.write("")
                st.markdown("**Estimated probability of passing**")
                st.progress(float(probability))
                st.markdown(f"### {probability:.1%}")
                st.caption(
                    "This is the model's estimated probability, "
                    "not a guarantee of the student's result."
                )
        else:
            st.markdown(
                '<div style="background:#F0F4ED; padding:24px; '
                'border-radius:14px;">'
                '<h3 style="color:#365744;">Ready to evaluate</h3>'
                '<p style="color:#526D5B;">Enter the student details '
                'and select Generate prediction to see the result.</p>'
                '</div>',
                unsafe_allow_html=True
            )

            st.write("")
            st.caption("The model uses two input features.")
            st.markdown("- Study hours per day")
            st.markdown("- Attendance percentage")

# ---------- DETAILS AFTER PREDICTION ----------
if predict_clicked and name.strip():
    student_data = [[hours, attendance]]
    prediction = model.predict(student_data)[0]
    probability = model.predict_proba(student_data)[0][1]
    label = "Pass" if prediction == 1 else "Fail"

    st.write("")
    st.subheader("Student overview")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Study time", f"{hours:g} hrs/day")
    metric2.metric("Attendance", f"{attendance:g}%")
    metric3.metric("Pass probability", f"{probability:.1%}")

    st.write("")
    with st.container(border=True):
        st.subheader("💡 Understanding this prediction")

        st.write(
            f"For **{name.strip()}**, the model received "
            f"**{hours:g} study hours per day** and "
            f"**{attendance:g}% attendance**."
        )

        if prediction == 1:
            st.write(
                f"The estimated pass probability is {probability:.1%}, "
                "which is at least 50%. The model therefore classifies "
                "this input as **Pass**."
            )
        else:
            st.write(
                f"The estimated pass probability is {probability:.1%}, "
                "which is below 50%. The model therefore classifies "
                "this input as **Fail**."
            )

        st.caption(
            "These inputs are evaluated together. This explanation "
            "describes the model's decision rule; it does not prove "
            "that either factor caused the predicted outcome."
        )

    with st.container(border=True):
        st.subheader("📈 Study hours vs. pass probability")
        st.write(
            "This chart changes study hours while keeping attendance "
            f"fixed at {attendance:g}%."
        )

        study_hours = [h / 2 for h in range(49)]
        chart_data = pd.DataFrame({
            "Study hours": study_hours,
            "Pass probability": model.predict_proba([
                [h, attendance] for h in study_hours
            ])[:, 1]
        }).set_index("Study hours")

        st.line_chart(
            chart_data,
            y="Pass probability",
            color="#78947B"
        )

# ---------- FOOTER ----------

st.divider()

st.markdown("""
<div style="
    width: 100%;
    text-align: center;
    padding: 12px 0 4px 0;
    color: #718078;
    font-size: 0.85rem;
">
    <div style="font-weight: 700; letter-spacing: 1px;">
        STUDENT SUCCESS PREDICTOR
    </div>
    <div style="margin-top: 6px;">
        MACHINE LEARNING PROJECT
    </div>
    <div style="margin-top: 12px; font-size: 0.8rem;">
        Educational prototype using a small sample dataset.
        Predictions are illustrative, not guarantees of academic performance.
    </div>
</div>
""", unsafe_allow_html=True)
