import streamlit as st
import joblib
import pandas as pd

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# THEME STATE
# ---------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
try:
    model = joblib.load("student_pass_model.joblib")
except FileNotFoundError:
    st.error("⚠ Model file `student_pass_model.joblib` not found. Please run `train_model.py` first.")
    st.stop()

# ---------------------------------------------------------
# COLOR PALETTES
# ---------------------------------------------------------
DARK = {
    "bg_grad":      "radial-gradient(circle at 20% 0%, #0f1a0f 0%, #0a0e0a 55%, #050705 100%)",
    "card_bg":      "#0d150d",
    "card_hover":   "#00ff88",
    "border":       "#1f3a1f",
    "text":         "#e6ffe6",
    "muted":        "#7a9a7a",
    "heading":      "#00ff88",
    "input_bg":     "#0f1a0f",
    "input_text":   "#00ff88",
    "input_border": "#1f3a1f",
    "primary":      "#00ff88",
    "primary_dark": "#00c96a",
    "primary_soft": "rgba(0, 255, 136, 0.15)",
    "glow":         "rgba(0, 255, 136, 0.35)",
    "glow_strong":  "rgba(0, 255, 136, 0.65)",
    "btn_text":     "#0a0e0a",
    "banner_bg":    "linear-gradient(135deg, #0d1f0d 0%, #0a170a 100%)",
    "stat_bg":      "#0d150d",
    "chart_bg":     "#0d150d",
    "fail":         "#ff4466",
    "fail_glow":    "rgba(255, 68, 102, 0.6)",
    "hr":           "#1f3a1f",
    "footer":       "#3d5a3d",
    "code_bg":      "#050a05",
}

LIGHT = {
    "bg_grad":      "radial-gradient(circle at 20% 0%, #f0fdf4 0%, #ffffff 55%, #ecfdf5 100%)",
    "card_bg":      "#ffffff",
    "card_hover":   "#059669",
    "border":       "#a7f3d0",
    "text":         "#064e3b",
    "muted":        "#6b7280",
    "heading":      "#047857",
    "input_bg":     "#ffffff",
    "input_text":   "#047857",
    "input_border": "#a7f3d0",
    "primary":      "#059669",
    "primary_dark": "#047857",
    "primary_soft": "rgba(5, 150, 105, 0.15)",
    "glow":         "rgba(5, 150, 105, 0.25)",
    "glow_strong":  "rgba(5, 150, 105, 0.45)",
    "btn_text":     "#ffffff",
    "banner_bg":    "linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)",
    "stat_bg":      "#ffffff",
    "chart_bg":     "#ffffff",
    "fail":         "#dc2626",
    "fail_glow":    "rgba(220, 38, 38, 0.4)",
    "hr":           "#a7f3d0",
    "footer":       "#94a3b8",
    "code_bg":      "#f8fafc",
}

C = DARK if st.session_state.theme == "dark" else LIGHT

# ---------------------------------------------------------
# THEME TOGGLE (Top Right)
# ---------------------------------------------------------
spacer, toggle_col = st.columns([3, 1])
with toggle_col:
    is_dark = st.toggle(
        "🌙 Dark Mode",
        value=(st.session_state.theme == "dark"),
        key="theme_toggle",
    )
    new_theme = "dark" if is_dark else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

# ---------------------------------------------------------
# DYNAMIC CSS
# ---------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{
        background: {C["bg_grad"]};
        color: {C["text"]};
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 780px;
    }}

    [data-testid="stToggle"] label p {{
        color: {C["muted"]} !important;
        font-family: 'Consolas', monospace;
        font-size: 0.85rem;
        letter-spacing: 1px;
    }}
    [data-testid="stToggle"] [role="switch"][aria-checked="true"] {{
        background-color: {C["primary"]} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {C["heading"]} !important;
        font-family: 'Consolas', 'Courier New', monospace;
        letter-spacing: 0.5px;
    }}
    h1 {{ text-shadow: 0 0 12px {C["glow"]}; }}

    p, label, .stMarkdown,
    .stTextInput label, .stNumberInput label {{
        color: {C["text"]} !important;
    }}

    .stTextInput input,
    .stNumberInput input {{
        background-color: {C["input_bg"]} !important;
        color: {C["input_text"]} !important;
        border: 1px solid {C["input_border"]} !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-family: 'Consolas', monospace;
        caret-color: {C["primary"]};
    }}
    .stTextInput input:focus,
    .stNumberInput input:focus {{
        border: 1px solid {C["primary"]} !important;
        box-shadow: 0 0 0 3px {C["primary_soft"]} !important;
        outline: none !important;
    }}
    .stNumberInput button {{
        background-color: {C["input_bg"]} !important;
        color: {C["primary"]} !important;
        border: 1px solid {C["input_border"]} !important;
    }}
    .stNumberInput button:hover {{
        background-color: {C["primary"]} !important;
        color: {C["btn_text"]} !important;
    }}

    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, {C["primary"]} 0%, {C["primary_dark"]} 100%) !important;
        color: {C["btn_text"]} !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 1px;
        padding: 12px 24px !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 0 20px {C["glow"]};
        transition: all 0.25s ease;
        text-transform: uppercase;
        font-family: 'Consolas', monospace;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 0 30px {C["glow_strong"]};
    }}

    [data-testid="stNotification"] {{
        background-color: {C["card_bg"]} !important;
        border: 1px solid {C["border"]} !important;
        border-left: 4px solid {C["primary"]} !important;
        color: {C["text"]} !important;
        border-radius: 10px;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {C["card_bg"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 14px !important;
        padding: 20px 24px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.15),
                    inset 0 0 0 1px {C["primary_soft"]};
        transition: all 0.25s ease;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: {C["card_hover"]} !important;
        box-shadow: 0 4px 32px {C["glow"]} !important;
    }}

    /* ---------- Expander (dropdown) ---------- */
    details[data-testid="stExpander"] {{
        background: {C["card_bg"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 12px !important;
        margin: 10px 0 !important;
        overflow: hidden;
    }}
    details[data-testid="stExpander"] summary {{
        background: {C["card_bg"]} !important;
        color: {C["primary"]} !important;
        font-family: 'Consolas', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        padding: 12px 16px !important;
        cursor: pointer;
    }}
    details[data-testid="stExpander"] summary:hover {{
        color: {C["heading"]} !important;
        background: {C["primary_soft"]} !important;
    }}
    details[data-testid="stExpander"][open] summary {{
        border-bottom: 1px solid {C["border"]};
    }}

    /* ---------- Result banner ---------- */
    .result-banner {{
        background: {C["banner_bg"]};
        border: 1px solid {C["primary"]};
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 40px {C["glow"]},
                    inset 0 0 24px {C["primary_soft"]};
    }}
    .result-banner .label {{
        color: {C["muted"]};
        font-size: 0.85rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .result-banner .name {{
        color: {C["text"]};
        font-size: 1.1rem;
        margin-bottom: 12px;
    }}
    .result-banner .verdict-pass {{
        color: {C["primary"]};
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 4px;
        text-shadow: 0 0 20px {C["glow_strong"]};
        font-family: 'Consolas', monospace;
    }}
    .result-banner .verdict-fail {{
        color: {C["fail"]};
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 4px;
        text-shadow: 0 0 20px {C["fail_glow"]};
        font-family: 'Consolas', monospace;
    }}

    /* ---------- Stat row ---------- */
    .stat-row {{ display: flex; gap: 12px; margin: 12px 0; }}
    .stat-box {{
        flex: 1;
        background: {C["stat_bg"]};
        border: 1px solid {C["border"]};
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }}
    .stat-box .k {{
        color: {C["muted"]};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    .stat-box .v {{
        color: {C["primary"]};
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'Consolas', monospace;
        margin-top: 4px;
    }}

    /* ---------- Why-this-result inner card ---------- */
    .why-card {{
        background: {C["card_bg"]};
        border: 1px solid {C["border"]};
        border-top: 3px solid {C["primary"]};
        border-radius: 10px;
        padding: 18px;
        margin: 8px 0;
    }}
    .why-intro {{
        color: {C["muted"]};
        font-family: 'Consolas', monospace;
        font-size: 0.85rem;
        margin: 0 0 14px 0;
        line-height: 1.6;
    }}
    .formula {{
        background: {C["code_bg"]};
        border-left: 3px solid {C["primary"]};
        border-radius: 8px;
        padding: 14px 18px;
        font-family: 'Consolas', monospace;
        font-size: 0.85rem;
        color: {C["primary"]};
        margin: 12px 0;
        overflow-x: auto;
        white-space: nowrap;
    }}
    .contrib-row {{
        display: flex;
        gap: 10px;
        margin: 14px 0;
    }}
    .contrib-box {{
        flex: 1;
        background: {C["stat_bg"]};
        border: 1px solid {C["border"]};
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
    }}
    .contrib-box .k {{
        color: {C["muted"]};
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }}
    .contrib-box .v {{
        color: {C["primary"]};
        font-size: 1.2rem;
        font-weight: 700;
        font-family: 'Consolas', monospace;
    }}
    .contrib-box .sub {{
        color: {C["muted"]};
        font-size: 0.68rem;
        font-family: 'Consolas', monospace;
        margin-top: 4px;
    }}
    .contrib-box.negative .v {{ color: {C["fail"]}; }}
    .contrib-box.winner {{
        border-color: {C["primary"]};
        box-shadow: 0 0 12px {C["glow"]};
    }}
    .final-score {{
        background: {C["banner_bg"]};
        border: 1px solid {C["primary"]};
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        font-family: 'Consolas', monospace;
        font-size: 1rem;
        color: {C["primary"]};
        font-weight: 700;
        letter-spacing: 1px;
        margin: 14px 0 0 0;
        box-shadow: 0 0 20px {C["glow"]};
    }}
    .reason-text {{
        color: {C["text"]};
        font-size: 0.95rem;
        line-height: 1.7;
        margin: 4px 0;
        padding: 14px 18px;
        background: {C["primary_soft"]};
        border-radius: 10px;
        border-left: 3px solid {C["primary"]};
    }}
    .reason-text b {{ color: {C["primary"]}; }}

    hr {{ border-color: {C["hr"]} !important; }}

    .stBarChart, [data-testid="stArrowVegaLiteChart"] {{
        background: {C["chart_bg"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 12px !important;
        padding: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(f"""
<div style="text-align:center; padding: 10px 0 20px 0;">
    <div style="color:{C['primary']}; font-family:Consolas,monospace; font-size:0.85rem;
                letter-spacing:6px; text-transform:uppercase; opacity:0.8;">
        ▸ ML Deployment ▸ Student Pass Predictor
    </div>
    <h1 style="margin: 8px 0 0 0;">🎓 Predictor Console</h1>
    <div style="color:{C['muted']}; font-family:Consolas,monospace; font-size:0.9rem;
                margin-top:6px;">
        Logistic Regression · Model Persistence Demo
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# INPUT CARD
# ---------------------------------------------------------
with st.container(border=True):
    st.markdown('<h3 style="margin-top:0;">▸ Student Input</h3>', unsafe_allow_html=True)

    student_name = st.text_input("Student Name", placeholder="e.g. Juan Dela Cruz")

    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0, max_value=24.0,
            value=5.0, step=0.5
        )
    with col2:
        attendance = st.number_input(
            "Attendance %",
            min_value=0.0, max_value=100.0,
            value=80.0, step=1.0
        )

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("▶  RUN PREDICTION")

if predict_clicked:
    if not student_name.strip():
        st.warning("⚠ Please enter the student name.")
    else:
        # ----- Predict -----
        features = [[study_hours, attendance]]
        prediction = int(model.predict(features)[0])
        verdict = "PASS" if prediction == 1 else "FAIL"
        css_class = "verdict-pass" if prediction == 1 else "verdict-fail"

        # ----- Extract model math -----
        coefs = model.coef_[0]
        intercept = model.intercept_[0]

        study_coef = coefs[0]
        att_coef   = coefs[1]

        study_contrib = study_coef * study_hours
        att_contrib   = att_coef   * attendance
        total_score   = study_contrib + att_contrib + intercept

        stronger_is_attendance = abs(att_contrib) > abs(study_contrib)
        stronger_name  = "Attendance" if stronger_is_attendance else "Study Hours"
        stronger_value = max(abs(att_contrib), abs(study_contrib))

        # ============================================
        # 1. RESULT BANNER
        # ============================================
        st.markdown(f"""
        <div class="result-banner">
            <div class="label">▸ Prediction Result</div>
            <div class="name">{student_name}</div>
            <div class="{css_class}">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)

        # ============================================
        # 2. STAT ROW
        # ============================================
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="k">Study Hours</div>
                <div class="v">{study_hours}</div>
            </div>
            <div class="stat-box">
                <div class="k">Attendance</div>
                <div class="v">{attendance}%</div>
            </div>
            <div class="stat-box">
                <div class="k">Class</div>
                <div class="v">{prediction}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ============================================
        # 3. WHY THIS RESULT? (COLLAPSIBLE DROPDOWN)
        # ============================================
        winner_study = "winner" if not stronger_is_attendance else ""
        winner_att   = "winner" if stronger_is_attendance     else ""

        with st.expander("▸  Why This Result?  (click to see the math)"):
            math_html = f"""<div class="why-card">
<h4 style="margin-top:0;">Why This Result?</h4>
<p class="why-intro">
Logistic Regression gives each input a <b>weight</b>, multiplies it,
sums everything up, and adds a <b>base offset</b>.
If the total score is <b>above 0</b>, the student is predicted to PASS.
</p>

<div class="formula">
score = ({study_coef:+.3f} × study_hours) + ({att_coef:+.3f} × attendance) + ({intercept:+.3f})
</div>

<div class="contrib-row">
<div class="contrib-box {winner_study}">
<div class="k">Study Hours</div>
<div class="v">{study_contrib:+.2f}</div>
<div class="sub">{study_coef:.3f} × {study_hours}</div>
</div>
<div class="contrib-box {winner_att}">
<div class="k">Attendance</div>
<div class="v">{att_contrib:+.2f}</div>
<div class="sub">{att_coef:.3f} × {attendance}</div>
</div>
<div class="contrib-box negative">
<div class="k">Base Offset</div>
<div class="v">{intercept:+.2f}</div>
<div class="sub">intercept</div>
</div>
</div>

<div class="final-score">
Final Score = {total_score:+.2f} &nbsp;→&nbsp; {verdict}
</div>
</div>"""
            st.markdown(math_html, unsafe_allow_html=True)

        # ============================================
        # 3b. WHY PASS OR FAIL? (COLLAPSIBLE DROPDOWN)
        # ============================================
        with st.expander("▸  Why Pass or Fail?  (plain English)"):
            if prediction == 1:
                reason = f"""<div class="reason-text">
✅ <b>{student_name}</b> was predicted to <b>PASS</b> because the combined score
(<b>{total_score:+.2f}</b>) is above the threshold of <b>0</b>.
<br><br>
🏆 <b>{stronger_name}</b> had the strongest positive influence on this decision,
contributing <b>{stronger_value:.2f}</b> to the score.
</div>"""
            else:
                reason = f"""<div class="reason-text">
❌ <b>{student_name}</b> was predicted to <b>FAIL</b> because the combined score
(<b>{total_score:+.2f}</b>) is at or below the threshold of <b>0</b>.
<br><br>
⚠ The inputs were not strong enough to overcome the model's base offset of
<b>{intercept:+.2f}</b>. <b>{stronger_name}</b> contributed the most
(<b>{stronger_value:.2f}</b>), but still not enough.
</div>"""
            st.markdown(reason, unsafe_allow_html=True)

        # ============================================
        # 4. HOW THIS PREDICTION WAS MADE
        # ============================================
        with st.container(border=True):
            st.markdown(f"""
            <h4 style="margin-top:0;">▸ How This Prediction Was Made</h4>
            <p style="color:{C['text']}; font-family:Consolas,monospace;
                      font-size:0.9rem; margin:0;">
                The saved model
                <span style="color:{C['primary']};">student_pass_model.joblib</span>
                was loaded with
                <span style="color:{C['primary']};">joblib.load()</span>.<br><br>
                Input features:
                <span style="color:{C['primary']};">study_hours={study_hours}</span>,
                <span style="color:{C['primary']};">attendance_percentage={attendance}</span><br><br>
                The model applied its learned coefficients to produce
                class <span style="color:{C['primary']};">{prediction}</span>
                ({verdict}). No retraining was performed.
            </p>
            """, unsafe_allow_html=True)

        # ============================================
        # 5. BONUS CHART
        # ============================================
        st.markdown('<h4>▸ Input vs Predicted Class</h4>', unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            "Metric": ["Study Hours", "Attendance %", "Class (0/1)"],
            "Value": [study_hours, attendance, prediction],
        })
        st.bar_chart(chart_df.set_index("Metric"), color=C["primary"])

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(f"""
<hr/>
<div style="text-align:center; color:{C['footer']}; font-family:Consolas,monospace;
            font-size:0.75rem; letter-spacing:2px; padding-top:10px;">
    ▸ TRAIN ONCE · SAVE · LOAD · DEPLOY ◂
</div>
""", unsafe_allow_html=True)