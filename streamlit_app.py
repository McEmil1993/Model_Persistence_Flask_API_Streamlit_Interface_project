from html import escape
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

import importlib
import student_model

# Refresh the model class retained by a running Streamlit process.
importlib.reload(student_model)

st.set_page_config(page_title="StudyTrack | Student Predictor", page_icon="🎓", layout="wide")
model = joblib.load(Path(__file__).resolve().parent / "student_pass_model.joblib")


def get_prediction_label(score):
    return "Pass" if score >= 60 else "Fail"


def build_score_chart(hours, scores, selected=None):
    """Draw scores without clipping estimates outside the usual 0–100 range."""
    low = np.floor(min(0, float(np.min(scores))) / 20) * 20
    high = np.ceil(max(100, float(np.max(scores))) / 20) * 20
    x = lambda h: 70 + float(h) / 20 * 630
    y = lambda s: 290 - (float(s) - low) / (high - low) * 240
    points = " ".join(f"{x(h):.2f},{y(s):.2f}" for h, s in zip(hours, scores))
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 350" role="img" aria-labelledby="chart-title chart-desc" style="width:100%;height:auto;font-family:Arial,sans-serif">',
        '<title id="chart-title">Study time and predicted score</title>',
        '<desc id="chart-desc">The teal line shows model scores from zero to twenty hours. The dashed line marks the passing score of sixty. A navy dot marks your submitted prediction.</desc>',
        '<defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#14b8a6" stop-opacity=".25"/><stop offset="100%" stop-color="#14b8a6" stop-opacity=".02"/></linearGradient></defs>',
        f'<rect x="70" y="50" width="630" height="{y(60)-50:.2f}" fill="#f0fdfa"/>',
    ]
    for score in np.linspace(low, high, 6):
        svg.append(f'<line x1="70" y1="{y(score):.2f}" x2="700" y2="{y(score):.2f}" stroke="#e2e8f0"/><text x="54" y="{y(score)+4:.2f}" text-anchor="end" fill="#64748b" font-size="12">{score:g}</text>')
    for hour in range(0, 21, 5):
        svg.append(f'<text x="{x(hour)}" y="315" text-anchor="middle" fill="#64748b" font-size="12">{hour}</text>')
    svg.extend([
        f'<polygon points="70,290 {points} 700,290" fill="url(#area)"/>',
        f'<line x1="70" y1="{y(60):.2f}" x2="700" y2="{y(60):.2f}" stroke="#d97706" stroke-dasharray="6 5"/>',
        f'<text x="700" y="{y(60)-9:.2f}" text-anchor="end" fill="#a16207" font-size="12">Passing score · 60</text>',
        f'<polyline points="{points}" fill="none" stroke="#0d9488" stroke-width="3"/>',
        '<text x="70" y="26" fill="#64748b" font-size="12">Predicted score</text>',
        '<text x="385" y="342" text-anchor="middle" fill="#64748b" font-size="12">Study time (hours)</text>',
    ])
    if selected:
        sx, sy = x(selected['hours']), y(selected['score'])
        label = f"{selected['hours']:g} hours · score {selected['score']:.2f} · {get_prediction_label(selected['score'])}"
        svg.append(f'<line x1="{sx}" y1="{sy}" x2="{sx}" y2="290" stroke="#94a3b8" stroke-dasharray="4 4"/><circle cx="{sx}" cy="{sy}" r="12" fill="#0f172a" fill-opacity=".12"/><circle cx="{sx}" cy="{sy}" r="6" fill="#0f172a" stroke="white" stroke-width="2"><title>{escape(label)}</title></circle>')
    return "".join(svg) + "</svg>"


st.markdown("""
<style>
.stApp { background: #f3f6fa; color: #17243b; }
.block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3, p, label { color: #17243b; }
[data-testid="stCaptionContainer"] { color: #64748b; }
[data-testid="stForm"] { background: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 24px; }
[data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #17243b !important; }
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { background: #f8fafc !important; color: #17243b !important; -webkit-text-fill-color: #17243b !important; }
[data-testid="stNumberInput"] button { background: #e2e8f0 !important; color: #17243b !important; }
[data-testid="stFormSubmitButton"] button { width: 100%; background: #0d9488; border: none; color: white; border-radius: 10px; min-height: 46px; }
[data-testid="stFormSubmitButton"] button:hover { background: #0f766e; color: white; }
.empty-state { background: #eaf0f7; border: 1px dashed #b6c5d6; border-radius: 16px; padding: 34px; margin-top: 16px; }
.empty-state h3 { margin: 0 0 8px; }
.empty-state p { color: #64748b; margin: 0; }
.result-banner { padding: 16px 20px; border-radius: 12px; margin-top: 16px; line-height: 1.6; }
.pass { background: #ccfbf1; color: #115e59; }
.fail { background: #fff7ed; color: #9a3412; }
.legend { display: flex; flex-wrap: wrap; gap: 20px; color: #64748b; font-size: 13px; margin-bottom: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.swatch { width: 16px; height: 3px; background: #0d9488; display: inline-block; }
.swatch.threshold { background: #d97706; }
.swatch.selected { width: 8px; height: 8px; border-radius: 50%; background: #0f172a; }
@media (max-width: 640px) { .block-container { padding-top: 1rem; } }
</style>
""", unsafe_allow_html=True)

form_column, result_column = st.columns([1, 1.65], gap="large")
with form_column:
    with st.form("student_form"):
        st.subheader("Your study plan")
        st.caption("Add your details to explore a prediction.")
        student_name = st.text_input("Student name", placeholder="e.g. nard nard")
        study_hours = st.number_input("Study time (hours)", min_value=0.0, max_value=20.0, step=0.5)
        attendance_percentage = st.number_input("Attendance percentage", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        submitted = st.form_submit_button("View my prediction", type="primary")
    st.caption("Explore study time from 0 to 20 hours.")

if submitted:
    st.session_state["student_prediction"] = {
        "name": student_name.strip() or "Student",
        "hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "score": float(model.predict(np.array([[study_hours, attendance_percentage]]))[0]),
    }

result = st.session_state.get("student_prediction")
if result and "attendance_percentage" not in result:
    result = None
chart_attendance = result["attendance_percentage"] if result else attendance_percentage
with result_column:
    st.subheader("Prediction overview")
    st.caption("A snapshot of your submitted study plan.")
    if result:
        st.metric("Attendance", f"{result['attendance_percentage']:g}%")
        predicted_class = get_prediction_label(result["score"])
        st.write("Results for", result["name"])
        hours_metric, score_metric, class_metric = st.columns(3)
        hours_metric.metric("Study time", f"{result['hours']:g} h")
        score_metric.metric("Predicted score", f"{result['score']:.2f}")
        class_metric.metric("Predicted result", predicted_class)
        message = ("Your study plan meets the model's passing threshold." if predicted_class == "Pass"
                   else "Your study plan is below the model's passing threshold. Try exploring more study time.")
        st.markdown(f'<div class="result-banner {predicted_class.lower()}" role="status">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state"><h3>Your next step starts here</h3><p>Enter your study time and select “View my prediction” to see your estimated score and result.</p></div>', unsafe_allow_html=True)

st.write("")
with st.container(border=True):
    st.subheader("The bigger picture")
    st.caption(f"Chart holds attendance at {chart_attendance:g}%.")
    st.caption("See how the model's predicted score changes with study time.")
    st.markdown('<div class="legend"><span><i class="swatch"></i>Predicted score</span><span><i class="swatch threshold"></i>Passing threshold</span><span><i class="swatch selected"></i>Your prediction</span></div>', unsafe_allow_html=True)
    hours_to_compare = np.linspace(0, 20, 81)
    comparison_scores = model.predict(np.column_stack([hours_to_compare, np.full_like(hours_to_compare, chart_attendance)]))
    st.markdown(build_score_chart(hours_to_compare, comparison_scores, result), unsafe_allow_html=True)
    if result:
        st.caption(f"Your selection: {result['hours']:g} hours · predicted score {result['score']:.2f} · {get_prediction_label(result['score'])}")

with st.expander("How to read your prediction"):
    st.write("The saved model estimates a score from study hours and attendance. Scores of 60 or higher are classified as Pass; lower scores are classified as Fail. The chart shows estimates across 0–20 hours, with your submitted study time marked in navy.")
    st.write("For learning purposes. Actual results depend on more than study time. The model uses study hours and attendance, trained on synthetic classroom data, and estimated scores may fall outside 0–100.")
