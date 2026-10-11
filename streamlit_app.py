from html import escape
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

import importlib
import student_model

# Refresh the model class retained by a running Streamlit process.
importlib.reload(student_model)

st.set_page_config(page_title="StudyTrack | Study planner", page_icon="📚", layout="wide")

def load_model():
    return joblib.load(Path(__file__).resolve().parent / "student_pass_model.joblib")

model = load_model()


def build_score_chart(selected_hours=None, attendance=80.0):
    hours = np.linspace(0, 20, 81)
    scores = model.predict(np.column_stack([hours, np.full_like(hours, attendance)]))
    lower = min(0, float(min(scores)))
    upper = max(100, float(max(scores))) + 10
    def x(h):
        return 65 + float(h) / 20 * 650
    def y(s):
        return 270 - (float(s) - lower) / (upper - lower) * 225
    points = " ".join(f"{x(h):.2f},{y(s):.2f}" for h, s in zip(hours, scores))
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 330" role="img" aria-labelledby="chart-title chart-desc" style="width:100%;height:auto;font-family:Arial,sans-serif">',
        '<title id="chart-title">Study hours and predicted score</title>',
        '<desc id="chart-desc">The purple line shows estimated scores from zero to twenty study hours. The dashed green line marks the passing score of sixty. A highlighted dot shows your latest prediction.</desc>',
        f'<rect x="65" y="45" width="650" height="{y(60)-45:.2f}" rx="6" fill="#edf8f3"/>',
    ]
    for score in np.linspace(lower, upper, 6):
        parts.append(f'<line x1="65" y1="{y(score):.2f}" x2="715" y2="{y(score):.2f}" stroke="#e3e9ef"/>')
        parts.append(f'<text x="53" y="{y(score)+4:.2f}" text-anchor="end" font-size="12" fill="#64748b">{score:.0f}</text>')
    for hour in range(0, 21, 5):
        parts.append(f'<text x="{x(hour)}" y="293" text-anchor="middle" font-size="12" fill="#64748b">{hour}</text>')
    parts.extend([
        f'<line x1="65" y1="{y(60):.2f}" x2="715" y2="{y(60):.2f}" stroke="#168568" stroke-dasharray="6 5"/>',
        f'<text x="710" y="{y(60)-9:.2f}" text-anchor="end" font-size="12" fill="#11664f">Passing score: 60</text>',
        f'<polyline points="{points}" fill="none" stroke="#5364df" stroke-width="3" stroke-linejoin="round"/>',
        '<text x="65" y="25" font-size="12" fill="#64748b">Predicted score</text>',
        '<text x="390" y="321" text-anchor="middle" font-size="12" fill="#64748b">Study hours</text>',
    ])
    if selected_hours is not None:
        score = float(model.predict([[selected_hours, attendance]])[0])
        parts.extend([
            f'<line x1="{x(selected_hours)}" y1="{y(score):.2f}" x2="{x(selected_hours)}" y2="270" stroke="#5364df" stroke-dasharray="4 4" opacity="0.5"/>',
            f'<circle cx="{x(selected_hours)}" cy="{y(score):.2f}" r="7" fill="#5364df" stroke="white" stroke-width="3"><title>Your prediction: {selected_hours:g} hours, score {score:.1f}</title></circle>',
        ])
    parts.append('</svg>')
    return "".join(parts)


st.markdown("""
<style>
.stApp { background: #f3f5fa; color: #18253e; }
.block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 3rem; }
h1, h2, h3, p, label { color: #18253e; }
[data-testid="stCaptionContainer"] { color: #62718a; }
[data-testid="stForm"] { background: white; border: 1px solid #e2e7f0; border-radius: 20px; padding: 24px; }
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
background: #f8fafc !important; color: #18253e !important; -webkit-text-fill-color: #18253e !important; }
[data-testid="stNumberInput"] button { background: #e8edf6 !important; color: #18253e !important; }
[data-testid="stWidgetLabel"] p { color: #18253e !important; }
[data-testid="stFormSubmitButton"] button { border-radius: 10px; background: #5364df; color: white; border: 0; }
[data-testid="stFormSubmitButton"] button p { color: white !important; }
.brand { color: #5364df; font-size: 13px; font-weight: 700; letter-spacing: 2px; margin-bottom: 16px; }
.result { background: white; border: 1px solid #e2e7f0; border-radius: 20px; padding: 24px; min-height: 225px; }
.eyebrow { font-size: 12px; color: #62718a; letter-spacing: 1px; text-transform: uppercase; }
.result h2 { margin: 8px 0 16px; font-size: 24px; }
.stats { display: flex; gap: 30px; flex-wrap: wrap; margin: 18px 0; }
.stat strong { display: block; font-size: 30px; color: #18253e; }
.stat span { font-size: 13px; color: #62718a; }
.badge { display: inline-block; border-radius: 30px; padding: 6px 14px; font-size: 13px; font-weight: 700; }
.pass { background: #e1f5eb; color: #11664f; }
.fail { background: #fff0da; color: #975814; }
.chart { background: white; border: 1px solid #e2e7f0; border-radius: 20px; padding: 20px 22px 8px; }
.chart h3 { margin: 0 0 4px; font-size: 20px; }
.chart p { margin: 0 0 14px; color: #62718a; font-size: 13px; }
</style>
<div class="brand">STUDYTRACK / STUDENT PLANNER</div>
""", unsafe_allow_html=True)

form_column, result_column = st.columns([1, 1.5], gap="large")
with form_column:
    with st.form("student_form"):
        st.subheader("Plan your study time")
        st.caption("Enter your details to explore a prediction.")
        student_name = st.text_input("Student name", placeholder="Enter name")
        study_hours = st.number_input("Study hours", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
        attendance_percentage = st.number_input("Attendance percentage", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        submitted = st.form_submit_button("See my prediction", type="primary", use_container_width=True)

if submitted:
    st.session_state["student_prediction"] = {
        "name": student_name.strip() or "Student",
        "hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "score": float(model.predict([[study_hours, attendance_percentage]])[0]),
    }

result = st.session_state.get("student_prediction")
if result and "attendance_percentage" not in result:
    result = None
chart_attendance = result["attendance_percentage"] if result else attendance_percentage
with result_column:
    if result:
        st.metric("Attendance", f"{result['attendance_percentage']:g}%")
        label = "Pass" if result["score"] >= 60 else "Fail"
        message = "Your study time meets the estimated passing threshold." if label == "Pass" else "Try a longer study session to explore how the estimate changes."
        st.markdown(f'<div class="result"><div class="eyebrow">Your latest prediction</div>'
                    f'<h2>{escape(result["name"])}&#39;s study outlook</h2>'
                    f'<span class="badge {label.lower()}">{label} · estimated result</span>'
                    f'<div class="stats"><div class="stat"><strong>{result["hours"]:g} h</strong><span>Study time</span></div>'
                    f'<div class="stat"><strong>{result["score"]:.1f}</strong><span>Predicted score</span></div>'
                    '<div class="stat"><strong>60</strong><span>Passing score</span></div></div>'
                    f'<p>{message}</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result"><div class="eyebrow">Your study outlook</div>'
                    '<h2>A little planning goes a long way.</h2><p>Submit your study time to see your estimated score and Pass or Fail result.</p>'
                    '<div class="stats"><div class="stat"><strong>60</strong><span>Passing score</span></div>'
                    '<div class="stat"><strong>0–20 h</strong><span>Explore study time</span></div></div></div>', unsafe_allow_html=True)

st.write("")
chart_column, info_column = st.columns([2.3, 1], gap="large")
with chart_column:
    st.markdown('<div class="chart"><h3>Your study trajectory</h3><p>Purple line: estimated score · Green area: passing range</p>'
                + build_score_chart(result["hours"] if result else None, chart_attendance) + '</div>', unsafe_allow_html=True)
    st.caption(f"Chart holds attendance at {chart_attendance:g}%.")
    st.caption("The highlighted dot marks your latest submitted prediction.")
with info_column:
    st.subheader("Read your outlook")
    st.write("Follow the purple line to compare study hours with estimated scores. The dashed green line marks the passing threshold of 60.")
    st.write("Change your study time and submit again to explore another result.")
    st.caption("This learning model uses study hours and attendance, trained on synthetic classroom data. Estimates may fall outside 0–100 and do not guarantee actual results.")
