from html import escape
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="StudyTrack | Student Insights", page_icon="🎓", layout="wide")
model = joblib.load(Path(__file__).resolve().parent / "student_pass_model.joblib")


def get_prediction_label(score):
    return "Pass" if score >= 60 else "Fail"


def build_score_chart(hours, scores, selected=None):
    """Keep the complete score range visible, including estimates above 100."""
    low = float(np.floor(min(0, float(np.min(scores))) / 20) * 20)
    high = float(np.ceil(max(100, float(np.max(scores))) / 20) * 20)
    rows = [{"hours": float(h), "score": float(s), "result": get_prediction_label(s)}
            for h, s in zip(hours, scores)]
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("hours:Q", title="Study time (hours)", scale=alt.Scale(domain=[0, 20]),
                axis=alt.Axis(values=[0, 5, 10, 15, 20], grid=False)),
        y=alt.Y("score:Q", title="Estimated score", scale=alt.Scale(domain=[low, high])),
    )
    passing = alt.Chart(alt.Data(values=[{"start": 0, "end": 20, "bottom": 60, "top": high}])).mark_rect(
        color="#10b981", opacity=0.05).encode(x="start:Q", x2="end:Q", y="bottom:Q", y2="top:Q")
    threshold = alt.Chart(alt.Data(values=[{"score": 60}])).mark_rule(
        color="#d97706", strokeDash=[6, 5]).encode(y="score:Q")
    layers = [passing, base.mark_area(color="#8b5cf6", opacity=0.10),
              base.mark_line(color="#7c3aed", strokeWidth=3).encode(tooltip=[
                  alt.Tooltip("hours:Q", title="Study hours", format=".2f"),
                  alt.Tooltip("score:Q", title="Estimated score", format=".2f"),
                  alt.Tooltip("result:N", title="Result")]), threshold]
    if selected is not None:
        choice = alt.Chart(alt.Data(values=[{"hours": selected["hours"], "score": selected["score"]}]))
        layers.extend([
            choice.mark_rule(color="#a78bfa", strokeDash=[4, 4]).encode(x="hours:Q"),
            choice.mark_circle(size=180, color="#7c3aed", opacity=1, stroke="white", strokeWidth=3).encode(
                x="hours:Q", y="score:Q", tooltip=[alt.Tooltip("hours:Q", title="Your hours"),
                alt.Tooltip("score:Q", title="Your score", format=".2f")]),
        ])
    return alt.layer(*layers).properties(height=320, description="Estimated score by study time; passing score is 60.").configure_view(
        stroke=None).configure_axis(labelColor="#77738c", titleColor="#77738c", gridColor="#efedf5",
                                   labelFontSize=12, titleFontSize=12, titlePadding=14)


st.markdown("""
<style>
.stApp { background:#f7f8fc; color:#24243b; }
.block-container { max-width:1240px; padding-top:1.7rem; padding-bottom:2rem; }
[data-testid="stHeader"] { background:transparent; }
h1,h2,h3,p,label { color:#24243b; }
[data-testid="stCaptionContainer"] { color:#75758c; }
[data-testid="stForm"] { background:white; border:1px solid #e7e8f0; border-radius:20px; padding:24px; }
[data-testid="stVerticalBlockBorderWrapper"] > div { background:white; border-color:#e7e8f0; border-radius:20px; }
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input { background:#f7f8fc !important; color:#24243b !important; -webkit-text-fill-color:#24243b !important; }
[data-testid="stNumberInput"] button { background:#eef0f7; color:#24243b; }
[data-testid="stFormSubmitButton"] button { width:100%; border:none; border-radius:12px; min-height:48px; background:#7c3aed; color:white; }
[data-testid="stFormSubmitButton"] button:hover { background:#6d28d9; color:white; }
.brand { display:flex; align-items:center; gap:12px; padding-bottom:22px; border-bottom:1px solid #e5e7f0; margin-bottom:28px; }
.brand-icon { background:#7c3aed; color:white; width:38px; height:38px; display:grid; place-items:center; border-radius:12px; font-size:22px; }
.brand strong { font-size:20px; letter-spacing:-.6px; }
.brand small { margin-left:auto; color:#77738c; font-size:12px; }
.hero { display:flex; justify-content:space-between; align-items:center; gap:24px; padding:32px 36px; margin-bottom:26px; border-radius:22px; background:linear-gradient(110deg,#27203f,#493177); }
.hero h1 { color:white; font-size:clamp(28px,4vw,38px); font-weight:700; letter-spacing:-1.2px; margin:8px 0 12px; padding:0; }
.hero p { color:#d7cfe9; max-width:580px; margin:0; font-size:15px; line-height:1.7; }
.eyebrow { font-size:11px; font-weight:700; letter-spacing:2px; color:#c4b5fd; }
.hero-art { width:150px; flex-shrink:0; }
.section-label { color:#8c869e; font-size:11px; font-weight:700; letter-spacing:1.7px; margin-bottom:8px; }
.result-top { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:18px; }
.result-top h3 { margin:0; padding:0; font-size:23px; }
.badge { background:#edf9f3; color:#177450; border-radius:30px; padding:7px 12px; font-size:12px; font-weight:600; }
.badge.fail { background:#fff4e7; color:#a76210; }
.stats { display:grid; grid-template-columns:1.3fr 1fr 1fr; gap:12px; margin:16px 0; }
.stat { background:#f7f8fc; border:1px solid #ececf4; border-radius:14px; padding:18px; }
.stat.primary { background:#f3efff; border-color:#e6ddfc; }
.stat-label { color:#77738c; font-size:12px; }
.stat-value { color:#302943; font-size:30px; font-weight:700; letter-spacing:-1px; margin:8px 0 2px; }
.stat.primary .stat-value { color:#7c3aed; }
.stat small { color:#8c869e; font-size:11px; }
.insight { background:#f0faf6; border-left:3px solid #10b981; border-radius:10px; padding:16px 18px; color:#22674d; font-size:14px; line-height:1.6; }
.insight.fail { background:#fff8ee; border-color:#f59e0b; color:#885418; }
.empty { padding:25px 8px; text-align:center; }
.empty-icon { font-size:32px; margin-bottom:10px; }
.empty h3 { font-size:23px; }
.empty p { color:#828096; max-width:370px; margin:0 auto; line-height:1.7; }
.legend { display:flex; gap:20px; flex-wrap:wrap; color:#77738c; font-size:12px; margin:14px 0 4px; }
.legend span { display:flex; align-items:center; gap:7px; }
.line { display:inline-block; width:18px; height:3px; background:#7c3aed; border-radius:2px; }
.line.threshold { background:transparent; height:0; border-top:2px dashed #d97706; }
.line.dot { width:8px; height:8px; border-radius:50%; }
.footer { text-align:center; color:#8c869e; font-size:12px; padding-top:22px; }
@media(max-width:640px) { .block-container { padding-top:1rem; } .hero { padding:24px; } .hero-art { display:none; } .brand small { display:none; } .stats { gap:8px; } .stat { padding:12px; } .stat-value { font-size:24px; } }
</style>
""", unsafe_allow_html=True)

st.markdown('''<div class="brand"><span class="brand-icon">↗</span><strong>StudyTrack<span style="color:#7c3aed">.</span></strong><small>Student performance explorer</small></div>
<div class="hero"><div><span class="eyebrow">A LITTLE PLANNING. A CLEARER PATH.</span><h1>Turn your effort into insight.</h1><p>Find your study rhythm. Explore your estimated score and see how a little more study time could change your plan.</p></div>
<svg class="hero-art" viewBox="0 0 160 130" aria-hidden="true"><rect x="5" y="5" width="150" height="120" rx="20" fill="#ffffff0c"/><path d="M25 100H135M25 75H135M25 50H135" stroke="#ffffff14"/><rect x="33" y="78" width="18" height="22" rx="5" fill="#a78bfa"/><rect x="65" y="56" width="18" height="44" rx="5" fill="#c4b5fd"/><rect x="97" y="31" width="18" height="69" rx="5" fill="#ede9fe"/><path d="M32 64L73 37L120 16" stroke="#34d399" stroke-width="3" fill="none"/><circle cx="120" cy="16" r="5" fill="#34d399"/></svg></div>''', unsafe_allow_html=True)

form_column, result_column = st.columns([1, 1.8], gap="large")
with form_column:
    with st.form("student_form"):
        st.markdown('<div class="section-label">01 / YOUR INPUTS</div>', unsafe_allow_html=True)
        st.subheader("Build your study plan")
        st.caption("Two small details to explore your potential.")
        student_name = st.text_input("Student name", placeholder="Enter your name")
        study_hours = st.number_input("Study time (hours)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
        attendance_percentage = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        submitted = st.form_submit_button("Explore my prediction →", type="primary")
    st.caption("Change your inputs and submit again to compare plans.")

if submitted:
    st.session_state["student_prediction"] = {
        "name": student_name.strip() or "Student", "hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "score": float(model.predict([[study_hours, attendance_percentage]])[0]),
    }
result = st.session_state.get("student_prediction")
if result and "attendance_percentage" not in result:
    result = None
chart_attendance = result["attendance_percentage"] if result else attendance_percentage

with result_column:
    with st.container(border=True):
        st.markdown('<div class="section-label">02 / YOUR OUTLOOK</div>', unsafe_allow_html=True)
        if result:
            status = get_prediction_label(result["score"])
            margin = result["score"] - 60
            st.markdown(f'''<div class="result-top"><h3>{escape(result["name"])}’s prediction</h3><span class="badge {status.lower()}">{'✓ Pass' if status == 'Pass' else 'Below passing score'}</span></div>
<div class="stats"><div class="stat primary"><div class="stat-label">Estimated score</div><div class="stat-value">{result['score']:.2f}</div><small>Passing score: 60</small></div><div class="stat"><div class="stat-label">Study time</div><div class="stat-value">{result['hours']:g}<span style="font-size:15px"> h</span></div><small>Your submitted plan</small></div><div class="stat"><div class="stat-label">Attendance</div><div class="stat-value">{result['attendance_percentage']:g}<span style="font-size:15px">%</span></div><small>Held fixed in the chart</small></div></div>''', unsafe_allow_html=True)
            message = (f"Your estimate is <strong>{margin:.2f} points above</strong> the passing threshold. Explore the chart to compare other study plans."
                       if margin >= 0 else f"Your estimate is <strong>{abs(margin):.2f} points below</strong> the passing threshold. Explore additional study time at the same attendance.")
            st.markdown(f'<div class="insight {status.lower()}" role="status">{message}</div>', unsafe_allow_html=True)
            if result["hours"] < 20:
                next_hours = min(20, result["hours"] + 1)
                next_score = float(model.predict([[next_hours, chart_attendance]])[0])
                st.caption(f"What if? At {next_hours:g} study hours, the estimate is {next_score:.2f} ({next_score - result['score']:+.2f} points).")
        else:
            st.markdown('''<div class="empty"><div class="empty-icon">◎</div><h3>Your next chapter starts here.</h3><p>Build your plan on the left and explore your prediction. Your score, attendance, and study insights will appear here.</p></div>''', unsafe_allow_html=True)
            st.markdown('''<div class="stats"><div class="stat primary"><div class="stat-label">Estimated score</div><div class="stat-value">—</div></div><div class="stat"><div class="stat-label">Study time</div><div class="stat-value">—</div></div><div class="stat"><div class="stat-label">Attendance</div><div class="stat-value">—</div></div></div>''', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="section-label">03 / EXPLORE THE POSSIBILITIES</div>', unsafe_allow_html=True)
    st.subheader("See where your study time takes you")
    st.caption(f"Attendance fixed at {chart_attendance:g}% · Hover over the curve to inspect an estimate.")
    st.markdown('<div class="legend"><span><i class="line"></i>Estimated score</span><span><i class="line threshold"></i>Passing score · 60</span>' + ('<span><i class="line dot"></i>Your submitted plan</span>' if result else '') + '</div>', unsafe_allow_html=True)
    hours_to_compare = np.linspace(0, 20, 81)
    comparison_scores = model.predict(np.column_stack([hours_to_compare, np.full_like(hours_to_compare, chart_attendance)]))
    st.altair_chart(build_score_chart(hours_to_compare, comparison_scores, result), use_container_width=True, theme=None)
    st.caption("Scores are model estimates and can exceed 100. The green region is at or above the passing score.")

with st.expander("About these estimates"):
    st.write("The model uses study hours and attendance, trained on synthetic classroom data. A score of 60 or higher is classified as Pass. The chart keeps attendance fixed at your submitted percentage; before submitting, it uses the default attendance input.")
    st.write("This is a learning demonstration. Actual student outcomes depend on many factors, and these estimates are not validated on real students.")
st.markdown('<div class="footer">StudyTrack · Plan with curiosity. Learn with purpose.</div>', unsafe_allow_html=True)
