import html

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, #f4f7fe 0%, #eef2fb 100%); }
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
.block-container { max-width: 780px; padding-top: 2.2rem; padding-bottom: 3rem; }

/* Hero */
.hero {
  background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 55%, #7c3aed 100%);
  border-radius: 22px; padding: 38px 36px; color: #fff; margin-bottom: 26px;
  box-shadow: 0 18px 40px -16px rgba(79,70,229,.55);
}
.hero .badge {
  display: inline-block; font-size: 12px; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; background: rgba(255,255,255,.18);
  padding: 5px 12px; border-radius: 999px; margin-bottom: 14px;
}
.hero h1 { font-size: 34px; font-weight: 800; margin: 0 0 8px 0; color: #fff; line-height: 1.15; }
.hero p  { font-size: 15.5px; margin: 0; opacity: .88; line-height: 1.55; }

/* Cards */
.card {
  background: #fff; border-radius: 18px; padding: 24px 26px; margin-bottom: 20px;
  border: 1px solid #e5eaf5; box-shadow: 0 8px 24px -14px rgba(30,58,138,.25);
}
.card h3 { font-size: 17px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0; }
.card p, .card li { color: #475569; font-size: 14.5px; line-height: 1.65; }
.section-label { font-size: 12px; font-weight: 700; letter-spacing: .09em;
                 text-transform: uppercase; color: #6366f1; margin-bottom: 10px; }

/* Form */
[data-testid="stForm"] {
  background: #fff; border: 1px solid #e5eaf5 !important; border-radius: 18px;
  padding: 26px 26px 16px 26px; box-shadow: 0 8px 24px -14px rgba(30,58,138,.25);
}
label p { font-weight: 600 !important; color: #1e293b !important; font-size: 14px !important; }
.stTextInput input, .stNumberInput input { border-radius: 10px !important; }
.stFormSubmitButton button {
  width: 100%; border: none; border-radius: 12px; padding: 12px 0; font-weight: 700;
  font-size: 15.5px; color: #fff; background: linear-gradient(135deg, #4f46e5, #7c3aed);
  box-shadow: 0 10px 22px -10px rgba(79,70,229,.8); transition: transform .15s ease;
}
.stFormSubmitButton button:hover { transform: translateY(-2px); color: #fff; }
.stFormSubmitButton button p { color: #fff !important; }

/* Result */
.result { border-radius: 18px; padding: 26px; margin-bottom: 20px; display: flex; gap: 18px; align-items: center; }
.result.pass { background: #ecfdf5; border: 1px solid #a7f3d0; }
.result.fail { background: #fef2f2; border: 1px solid #fecaca; }
.result .icon { font-size: 38px; }
.result .title { font-size: 21px; font-weight: 800; margin: 0; }
.result.pass .title { color: #047857; } .result.fail .title { color: #b91c1c; }
.result .sub { margin: 4px 0 0 0; color: #475569; font-size: 14.5px; }

/* Metrics */
.metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 20px; }
.metric { background: #fff; border: 1px solid #e5eaf5; border-radius: 16px; padding: 16px 18px; }
.metric .k { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: #94a3b8; }
.metric .v { font-size: 26px; font-weight: 800; color: #1e293b; margin-top: 4px; }

/* Gauge */
.gauge { position: relative; height: 14px; border-radius: 999px; background: #e2e8f0; margin: 14px 0 8px 0; overflow: visible; }
.gauge .fill { height: 100%; border-radius: 999px; }
.gauge .mid { position: absolute; left: 50%; top: -5px; width: 2px; height: 24px; background: #64748b; }
.gauge-labels { display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; }
.foot { text-align: center; color: #94a3b8; font-size: 12.5px; margin-top: 28px; }
</style>
""", unsafe_allow_html=True)


# ---------- Model ----------
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")  # saved model, no retraining


model = load_model()

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <span class="badge">Machine Learning · Logistic Regression</span>
  <h1>🎓 Student Pass Predictor BY:JEFFERSON FERNANDEZ</h1>
  <p>Enter a student's daily study hours and get an instant prediction of whether they are
  likely to pass, powered by a pre-trained model.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Input form ----------
with st.form("predict_form"):
    st.markdown('<div class="section-label">Student information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    name = c1.text_input("Student name", placeholder="e.g. Juan Dela Cruz")
    hours = c2.number_input("Study hours per day", min_value=0.0, max_value=24.0,
                            value=3.0, step=0.5)
    submitted = st.form_submit_button("Predict result")

# ---------- Prediction ----------
if submitted:
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        X = pd.DataFrame({"study_hours": [hours]})
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
        safe_name = html.escape(name.strip())
        passed = pred == 1
        color = "#10b981" if passed else "#ef4444"

        # Result banner
        st.markdown(
            f'<div class="result {"pass" if passed else "fail"}">'
            f'<div class="icon">{"✅" if passed else "❌"}</div>'
            f'<div><p class="title">{safe_name} is predicted to {"PASS" if passed else "FAIL"}</p>'
            f'<p class="sub">Estimated pass probability: <b>{prob:.0%}</b></p></div></div>',
            unsafe_allow_html=True)

        # Metrics
        st.markdown(
            f'<div class="metrics">'
            f'<div class="metric"><div class="k">Study hours</div><div class="v">{hours:g}</div></div>'
            f'<div class="metric"><div class="k">Pass probability</div><div class="v">{prob:.0%}</div></div>'
            f'<div class="metric"><div class="k">Prediction</div>'
            f'<div class="v" style="color:{color}">{"Pass" if passed else "Fail"}</div></div></div>',
            unsafe_allow_html=True)

        # Gauge
        st.markdown(
            f'<div class="card"><h3>Confidence meter</h3>'
            f'<div class="gauge"><div class="fill" style="width:{prob*100:.1f}%;background:{color}"></div>'
            f'<div class="mid"></div></div>'
            f'<div class="gauge-labels"><span>0% Fail</span><span>50% threshold</span><span>100% Pass</span></div></div>',
            unsafe_allow_html=True)

        # Explanation
        st.markdown(
            f'<div class="card"><h3>How was this prediction made?</h3>'
            f'<p>The app loaded a Logistic Regression model that was already trained and saved in '
            f'<code>student_pass_model.joblib</code>. It took <b>{hours:g} study hours</b> as input and '
            f'calculated a pass probability of <b>{prob:.0%}</b>. If the probability is 50% or higher, '
            f'the predicted class is Pass; otherwise it is Fail. The model was not retrained.</p></div>',
            unsafe_allow_html=True)

        # Chart
        curve = pd.DataFrame({"study_hours": [h / 2 for h in range(0, 33)]})
        curve["pass_probability"] = model.predict_proba(curve[["study_hours"]])[:, 1]

        x_enc = alt.X("study_hours:Q", title="Study hours per day", scale=alt.Scale(domain=[0, 16]))
        y_enc = alt.Y("pass_probability:Q", title="Pass probability",
                      scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"))
        area = alt.Chart(curve).mark_area(color="#6366f1", opacity=0.12).encode(x=x_enc, y=y_enc)
        line = alt.Chart(curve).mark_line(color="#4f46e5", strokeWidth=3).encode(x=x_enc, y=y_enc)
        rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
            color="#94a3b8", strokeDash=[6, 4]).encode(y="y:Q")
        student = pd.DataFrame({"study_hours": [hours], "pass_probability": [prob]})
        point = alt.Chart(student).mark_point(size=220, filled=True, color=color,
                                              stroke="white", strokeWidth=2).encode(
            x="study_hours:Q", y="pass_probability:Q",
            tooltip=[alt.Tooltip("study_hours:Q", title="Hours"),
                     alt.Tooltip("pass_probability:Q", title="Probability", format=".0%")])
        chart = (area + line + rule + point).properties(height=320).configure_view(strokeWidth=0)

        st.markdown('<div class="card" style="padding-bottom:8px"><h3>Study hours vs. predicted pass probability</h3>'
                    '<p>The dot shows where this student sits on the curve.</p></div>',
                    unsafe_allow_html=True)
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"{name.strip()}: {hours:g} hours → predicted class: {'Pass' if passed else 'Fail'}")

st.markdown('<div class="foot">Built with Streamlit · scikit-learn · joblib</div>', unsafe_allow_html=True)