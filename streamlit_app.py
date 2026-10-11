from pathlib import Path
import importlib

import joblib
import numpy as np
import streamlit as st

import student_model

# Streamlit reruns this script but can retain an older imported model module.
# Reload before joblib resolves the class stored in the saved model.
importlib.reload(student_model)

st.set_page_config(page_title="StudyTrack | Student Predictor", page_icon="🎓", layout="wide")

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"
model = joblib.load(MODEL_PATH)


def get_prediction_label(score: float) -> str:
    return "Pass" if score >= 60 else "Fail"


def build_class_chart(study_hours, predicted_classes, selected_hours=None, selected_class=None):
    """Draw class predictions with SVG, without a dataframe dependency."""
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 380" '
        'role="img" aria-labelledby="chart-title chart-description" '
        'style="display:block;width:100%;height:auto;background:white;border:1px solid #e2e8f0;'
        'border-radius:20px;box-shadow:0 8px 28px rgba(15,23,42,.05);font-family:Inter,system-ui,sans-serif">',
        '<title id="chart-title">Study hours and predicted Pass or Fail class</title>',
        '<desc id="chart-description">Dots show predictions at half-hour intervals from 0 to 20 hours. '
        'The upper row is Pass and the lower row is Fail.</desc>',
        '<defs>'
        '<linearGradient id="chart-header" x2="1" y2="1"><stop stop-color="#eef2ff"/><stop offset="1" stop-color="#f8fafc"/></linearGradient>'
        '<linearGradient id="pass-lane"><stop stop-color="#ecfdf5"/><stop offset="1" stop-color="#d1fae5"/></linearGradient>'
        '<linearGradient id="fail-lane"><stop stop-color="#fff7ed"/><stop offset="1" stop-color="#ffedd5"/></linearGradient>'
        '</defs>',
        '<path d="M20 0H660Q680 0 680 20V90H0V20Q0 0 20 0" fill="url(#chart-header)"/>',
        '<rect x="26" y="22" width="4" height="42" rx="2" fill="#6366f1"/>',
        '<text x="42" y="40" fill="#1e293b" font-size="19" font-weight="700">Study time, visualized</text>',
        '<text x="42" y="63" fill="#64748b" font-size="12">Explore your predicted outcome at every half hour</text>',
        '<rect x="548" y="27" width="106" height="28" rx="14" fill="white" stroke="#e0e7ff"/>',
        '<circle cx="563" cy="41" r="4" fill="#6366f1"/>',
        '<text x="576" y="45" fill="#475569" font-size="10" font-weight="600">MODEL VIEW</text>',
        '<g transform="translate(0 100)">',
    ]
    for label, y, background, color in [("Pass", 65, "#ecfdf5", "#059669"), ("Fail", 185, "#fffbeb", "#d97706")]:
        gradient = "pass-lane" if label == "Pass" else "fail-lane"
        elements.append(f'<rect x="95" y="{y - 27}" width="550" height="54" rx="12" fill="url(#{gradient})"/>')
        elements.append(f'<rect x="95" y="{y - 17}" width="3" height="34" rx="1.5" fill="{color}"/>')
        elements.append(f'<line x1="105" y1="{y}" x2="635" y2="{y}" stroke="{color}" stroke-opacity=".15"/>')
        elements.append(f'<text x="82" y="{y + 5}" text-anchor="end" fill="{color}" font-size="13" font-weight="600">{label}</text>')
    for hours in range(0, 21, 5):
        x = 95 + hours / 20 * 550
        elements.append(f'<line x1="{x}" y1="35" x2="{x}" y2="215" stroke="#cbd5e1" stroke-opacity=".55" stroke-dasharray="3 6"/>')
        elements.append(f'<text x="{x}" y="237" text-anchor="middle" fill="#64748b" font-size="12">{hours}</text>')
    for hours, predicted_class in zip(study_hours, predicted_classes):
        x = 95 + float(hours) / 20 * 550
        y = 65 if predicted_class == 1 else 185
        color = "#059669" if predicted_class == 1 else "#d97706"
        label = "Pass" if predicted_class == 1 else "Fail"
        elements.append(f'<circle cx="{x}" cy="{y}" r="7" fill="{color}" fill-opacity=".09"/>')
        elements.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}" stroke="white" stroke-width="1.5">'
                        f'<title>{float(hours):g} study hours: {label}</title></circle>')
    if selected_hours is not None:
        x = 95 + float(selected_hours) / 20 * 550
        y = 65 if selected_class == "Pass" else 185
        elements.append(f'<line x1="{x}" y1="35" x2="{x}" y2="215" stroke="#6366f1" stroke-width="1.5" stroke-dasharray="4 5"/>')
        elements.append(f'<circle cx="{x}" cy="{y}" r="15" fill="#3b82f6" fill-opacity=".15"/>')
        badge_x = min(max(x - 48, 95), 549)
        elements.append(f'<rect x="{badge_x}" y="{y - 36}" width="96" height="23" rx="8" fill="#4f46e5"/>')
        elements.append(f'<text x="{badge_x + 48}" y="{y - 20}" text-anchor="middle" fill="white" font-size="11" font-weight="600">You: {float(selected_hours):g} h</text>')
        elements.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#2563eb" stroke="white" stroke-width="3">'
                        f'<title>Your prediction: {float(selected_hours):g} study hours, {selected_class}</title></circle>')
    elements.append('<text x="370" y="265" text-anchor="middle" fill="#475569" font-size="12" font-weight="500">Study time (hours)</text></g></svg>')
    return "".join(elements)


st.markdown("""
<style>
.stApp { background: #f5f7fb; color: #17243b; }
.block-container { max-width: 1150px; padding-top: 2.5rem; }
h1, h2, h3 { color: #17243b; }
[data-testid="stForm"] { background: white; border-radius: 16px; padding: 24px; }
[data-testid="stMetric"] { background: white; border: 1px solid #dce3ee; border-radius: 12px; padding: 16px; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #17243b !important; }
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #17243b !important;
    -webkit-text-fill-color: #17243b !important;
    caret-color: #17243b;
}
[data-testid="stNumberInput"] input { font-size: 1.15rem; font-weight: 600; }
[data-testid="stNumberInput"] button {
    background-color: #e8eef7 !important;
    color: #17243b !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label { color: #17243b !important; }
.prediction-fail {
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
    border-radius: 10px;
    padding: 16px 20px;
    font-weight: 600;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

st.caption("STUDYTRACK · STUDENT SUCCESS")
st.title("Small steps. Better study habits.")
st.write("Explore how study time relates to a student's predicted result.")
st.divider()

form_column, result_column = st.columns([1, 1.6], gap="large")
with form_column:
    with st.form("student_form"):
        st.subheader("Student details")
        st.caption("Enter a name, study time, and attendance to get started.")
        student_name = st.text_input("Student name", placeholder="e.g. Alex Santos")
        study_hours = st.number_input("Study hours", min_value=0.0, max_value=20.0, step=0.5)
        attendance_percentage = st.number_input(
            "Attendance percentage", min_value=0.0, max_value=100.0, value=80.0, step=1.0,
            help="Attendance must be between 0 and 100 percent. Attendance influences the predicted score.",
        )
        submitted = st.form_submit_button("Generate prediction", type="primary")
    st.caption("For learning purposes. Actual results depend on more than study time.")

if submitted:
    st.session_state["student_prediction"] = {
        "name": student_name.strip() or "Student",
        "hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "score": float(model.predict(np.array([[study_hours, attendance_percentage]]))[0]),
    }

with result_column:
    st.subheader("Prediction overview")
    if "student_prediction" in st.session_state and "attendance_percentage" in st.session_state["student_prediction"]:
        result = st.session_state["student_prediction"]
        predicted_class = get_prediction_label(result["score"])
        st.write(f"Results for **{result['name']}**")
        hours_metric, score_metric, class_metric = st.columns(3)
        hours_metric.metric("Study time", f"{result['hours']:g} hours")
        if "attendance_percentage" in result:
            st.metric("Attendance", f"{result['attendance_percentage']:g}%")
        score_metric.metric("Predicted score", f"{result['score']:.2f}")
        class_metric.metric("Predicted class", predicted_class)
        if predicted_class == "Pass":
            st.success("This study time meets the model's predicted passing threshold.")
        else:
            st.markdown(
                '<div class="prediction-fail" role="status">'
                "This study time falls below the model's predicted passing threshold."
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Your prediction will appear here after you submit the student details.")

st.divider()
chart_column, explanation_column = st.columns([1.6, 1], gap="large")
with explanation_column:
    st.subheader("How the prediction works")
    st.write(
        "The app loads a model trained on synthetic classroom examples using study hours and attendance. "
        "It feeds your study time and attendance into that model to estimate a score, then assigns Pass for scores "
        "of 60 or higher and Fail for lower scores."
    )
    st.caption("The saved model retains its learned values, so each prediction does not require retraining.")
    st.caption("Attendance affects predictions. This demonstration uses synthetic data, not measured student outcomes. Scores are estimates and may fall outside 0–100.")

with chart_column:
    st.subheader("Study time and predicted class")
    hours_to_compare = np.linspace(0, 20, 41)
    selected_result = st.session_state.get("student_prediction")
    chart_attendance = selected_result.get("attendance_percentage", attendance_percentage) if selected_result else attendance_percentage
    comparison_scores = model.predict(np.column_stack([hours_to_compare, np.full_like(hours_to_compare, chart_attendance)]))
    comparison_classes = (comparison_scores >= 60).astype(int)
    selected_result = st.session_state.get("student_prediction")
    chart_svg = build_class_chart(
        hours_to_compare,
        comparison_classes,
        selected_hours=selected_result["hours"] if selected_result else None,
        selected_class=get_prediction_label(selected_result["score"]) if selected_result else None,
    )
    st.markdown(chart_svg, unsafe_allow_html=True)
    st.caption(f"Chart holds attendance at {chart_attendance:g}%.")
    st.caption("Predicted class: 0 = Fail, 1 = Pass. Scores of 60 or higher are classified as Pass.")
    st.caption("Dots show predictions every 0.5 hours. A blue marker and label highlight your submitted study time.")

