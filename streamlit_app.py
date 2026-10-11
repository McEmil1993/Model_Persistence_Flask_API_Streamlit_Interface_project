"""Exercise 3 - Streamlit interface (uses the study-hours-only model)."""
import joblib
import numpy as np
import streamlit as st

BLUE = "#1E5EFF"
DARK = "#0B2447"
PASS_GREEN = "#1E9E6A"
FAIL_RED = "#D64545"

st.set_page_config(page_title="Student Pass Predictor", page_icon="", layout="centered")

# ---------- Blue & white styling ----------
st.markdown(
    f"""
    <style>
    .stApp {{ background: #FFFFFF; }}
    .hero {{
        background: linear-gradient(135deg, #1E5EFF 0%, #4C8DFF 100%);
        color: white; padding: 28px 30px; border-radius: 18px;
        margin-bottom: 22px; box-shadow: 0 8px 24px rgba(30,94,255,0.25);
    }}
    .hero h1 {{ color: white; margin: 0; font-size: 2rem; padding: 0; }}
    .hero p {{ margin: 6px 0 0 0; opacity: 0.92; font-size: 1rem; }}
    .card {{
        background: #F4F8FF; border: 1px solid #D6E4FF; border-radius: 14px;
        padding: 18px 20px; margin: 12px 0; color: {DARK};
    }}
    .result {{
        border-radius: 14px; padding: 20px 22px; margin: 16px 0;
        color: white; font-size: 1.15rem; font-weight: 600;
    }}
    .result.pass {{ background: {PASS_GREEN}; }}
    .result.fail {{ background: {FAIL_RED}; }}
    div.stButton > button {{
        background: {BLUE}; color: white; border: none; border-radius: 10px;
        padding: 0.6rem 1.6rem; font-weight: 600; width: 100%;
    }}
    div.stButton > button:hover {{ background: #0F47CC; color: white; }}
    [data-testid="stMetric"] {{
        background: #EEF4FF; border: 1px solid #D6E4FF;
        border-radius: 14px; padding: 14px 18px;
    }}
    </style>
    <div class="hero">
        <h1>Student Pass/Fail Predictor</h1>
        <p>Enter a student's study hours & attendance percentage and let the saved model predict the result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved model once (no retraining) and reuse it on every rerun."""
    return joblib.load("student_pass_model_v2.joblib")

# ---------- Inputs ----------
name = st.text_input("Student name", placeholder="e.g. Maria Santos")
col1, col2 = st.columns(2)
with col1:
    hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
with col2:
    attendance = st.number_input("Attendance percentage (%)", min_value=0.0,
                                 max_value=100.0, value=85.0, step=1.0)

predict_clicked = st.button("Predict")

if predict_clicked:
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        import plotly.graph_objects as go

        model = load_model()
        pred = model.predict([[hours, attendance]])[0]
        prob = model.predict_proba([[hours, attendance]])[0][1]
        label = "Pass" if pred == 1 else "Fail"

        # ---------- Result message ----------
        css = "pass" if pred == 1 else "fail"
        icon = "✅" if pred == 1 else "❌"
        st.markdown(
            f'<div class="result {css}">{icon} {name} is predicted to '
            f"{label.upper()} with {hours} study hours and {attendance:g}% attendance.</div>",
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        m1.metric("Chance of passing", f"{prob:.1%}")
        m2.metric("Predicted class", label)

        # ---------- Short explanation ----------
        st.markdown(
            f"""
            <div class="card">
            <b>For {name}, it simply loaded that saved file and weighed {hours} study hours
            together with {attendance:g}% attendance against the pattern it learned.
            More study hours and higher attendance generally mean a higher chance of
            passing. The model rates {name}'s chance at <b>{prob:.1%}</b>, and because
            a chance above 50% counts as a pass, the final prediction is <b>{label}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- Bonus chart (interactive, modern) ----------
        xs = np.linspace(0, 12, 200)
        # curve for THIS student's attendance, varying only the study hours
        grid = np.column_stack([xs, np.full_like(xs, attendance)])
        probs = model.predict_proba(grid)[:, 1]
        dot = PASS_GREEN if pred == 1 else FAIL_RED
        # study hours where the model flips from Fail to Pass
        if probs[0] > 0.5:
            cross = 0.0
        elif probs[-1] <= 0.5:
            cross = 12.0
        else:
            cross = float(xs[np.argmax(probs > 0.5)])

        fig = go.Figure()
        # soft background zones for Fail / Pass
        fig.add_vrect(x0=0, x1=cross, fillcolor=FAIL_RED, opacity=0.06, line_width=0)
        fig.add_vrect(x0=cross, x1=12, fillcolor=PASS_GREEN, opacity=0.07, line_width=0)
        fig.add_annotation(x=cross / 2, y=1.07, text="FAIL ZONE", showarrow=False,
                           font=dict(color=FAIL_RED, size=11))
        fig.add_annotation(x=(cross + 12) / 2, y=1.07, text="PASS ZONE", showarrow=False,
                           font=dict(color=PASS_GREEN, size=11))
        # smooth probability curve with gradient-like fill
        fig.add_trace(go.Scatter(
            x=xs, y=probs, mode="lines", name="Pass probability",
            line=dict(color=BLUE, width=4, shape="spline"),
            fill="tozeroy", fillcolor="rgba(30,94,255,0.14)",
            hovertemplate="<b>%{x:.1f} hrs</b><br>Pass chance: %{y:.1%}<extra></extra>",
        ))
        # threshold line
        fig.add_hline(y=0.5, line_dash="dot", line_color="#8EA2C4", line_width=1.5,
                      annotation_text="50% threshold", annotation_position="bottom right",
                      annotation_font=dict(color="#6B7C99", size=11))
        # drop line to the x-axis
        fig.add_shape(type="line", x0=hours, x1=hours, y0=0, y1=prob,
                      line=dict(color=dot, width=2, dash="dash"))
        # the student
        fig.add_trace(go.Scatter(
            x=[hours], y=[prob], mode="markers+text", name=f"{name}: {label}",
            text=[f"  {name} · {prob:.1%}"], textposition="top right",
            textfont=dict(color=DARK, size=13),
            marker=dict(size=18, color=dot, line=dict(color="white", width=3)),
            hovertemplate=f"<b>{name}</b><br>%{{x}} hrs<br>Pass chance: %{{y:.1%}}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=f"<b>Study hours vs predicted class</b><br><span style='font-size:12px;color:#6B7C99'>curve shown at {attendance:g}% attendance</span>",
                       font=dict(size=18, color=DARK)),
            template="plotly_white", height=430,
            margin=dict(l=10, r=10, t=90, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=DARK),
            xaxis=dict(title="Study hours", range=[0, 12], showgrid=False,
                       zeroline=False, showline=True, linecolor="#C9D7F2"),
            yaxis=dict(title="Probability of passing", range=[0, 1.12],
                       tickformat=".0%", gridcolor="#E6EEFC", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5,
                        xanchor="center"),
            hoverlabel=dict(bgcolor="white", font_size=13),
        )
        st.plotly_chart(fig, width="stretch")