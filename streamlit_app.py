"""Exercise 3 - Streamlit interface (pink pastel theme, uses student_pass_model.joblib)."""
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILE = "student_pass_model.joblib"

PINK = "#ff5fa2"
SOFT_PINK = "#ffd6e8"
LAVENDER = "#c9b6ff"
PLUM = "#5b2a48"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


st.set_page_config(page_title="Student Pass Predictor", page_icon="🌸", layout="centered")

# ---------- Custom styling ----------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500;600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Poppins', sans-serif;
        color: {PLUM};
    }}
    .stApp {{
        background: linear-gradient(160deg, #fff5f9 0%, #ffe9f3 45%, #f1e8ff 100%);
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 2.5rem; max-width: 720px; }}

    .hero {{
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
    }}
    .hero h1 {{
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        margin: 0;
        background: linear-gradient(90deg, {PINK}, #b983ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero p {{ color: #9a5a7d; margin-top: .4rem; font-weight: 300; }}

    .card {{
        background: rgba(255,255,255,.75);
        border: 1.5px solid {SOFT_PINK};
        border-radius: 22px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 10px 30px rgba(255, 95, 162, .12);
        margin-bottom: 1.2rem;
    }}
    .card h3 {{
        font-family: 'Playfair Display', serif;
        margin-top: 0;
        color: {PLUM};
    }}

    .result {{
        text-align: center;
        border-radius: 24px;
        padding: 1.6rem 1rem;
        margin: 1rem 0 1.2rem 0;
        box-shadow: 0 10px 30px rgba(185, 131, 255, .18);
    }}
    .result.pass {{ background: linear-gradient(135deg, #ffe0f0, #e9dcff); border: 2px solid {PINK}; }}
    .result.fail {{ background: linear-gradient(135deg, #fff0e6, #ffe0ea); border: 2px solid #ff9db8; }}
    .result .big {{ font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; }}
    .result .sub {{ font-weight: 300; color: #9a5a7d; margin-top: .3rem; }}

    .pill {{
        display: inline-block;
        background: {SOFT_PINK};
        color: {PLUM};
        padding: .25rem .9rem;
        border-radius: 999px;
        font-size: .85rem;
        margin-top: .6rem;
    }}

    /* Inputs */
    .stTextInput input, .stNumberInput input {{
        border-radius: 14px !important;
        border: 1.5px solid {SOFT_PINK} !important;
        background: #fffafc !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: {PINK} !important;
        box-shadow: 0 0 0 3px rgba(255,95,162,.18) !important;
    }}
    label p {{ font-weight: 500 !important; color: {PLUM} !important; }}

    /* Button */
    .stButton > button {{
        width: 100%;
        border: none;
        border-radius: 999px;
        padding: .75rem 1rem;
        font-weight: 600;
        font-size: 1.05rem;
        color: white;
        background: linear-gradient(90deg, {PINK}, #b983ff);
        box-shadow: 0 8px 20px rgba(255,95,162,.35);
        transition: transform .15s ease, box-shadow .15s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(255,95,162,.45);
        color: white;
    }}
    .stButton > button:active {{ transform: translateY(0); }}

    .footer {{ text-align:center; color:#b98aa5; font-size:.8rem; margin:2rem 0 .5rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🌸 Student Pass Predictor</h1>
        <p>✨ Tell me how much you study and I'll predict your result ✨</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Input card ----------
st.markdown('<div class="card"><h3>💌 Student details</h3>', unsafe_allow_html=True)
name = st.text_input("Student name", placeholder="e.g. Maria Santos")
hours = st.number_input(
    "Study hours per day 📚", min_value=0.0, max_value=24.0, value=5.0, step=0.5
)
predict_clicked = st.button("✨ Predict my result")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Prediction ----------
if predict_clicked:
    if not name.strip():
        st.warning("Please enter the student's name 💕")
    else:
        X = pd.DataFrame({"study_hours": [hours]})
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        # Result message
        if pred == 1:
            st.markdown(
                f"""
                <div class="result pass">
                    <div class="big">🎉 Yay, {name}!</div>
                    <div class="sub">You're predicted to <b>PASS</b></div>
                    <span class="pill">Pass probability: {prob:.0%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.balloons()
        else:
            st.markdown(
                f"""
                <div class="result fail">
                    <div class="big">🌷 Keep going, {name}!</div>
                    <div class="sub">You're predicted to <b>FAIL</b> for now. A little more study can change this.</div>
                    <span class="pill">Pass probability: {prob:.0%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Explanation
        st.markdown(
            f"""
            <div class="card">
                <h3>💡 How was this prediction generated?</h3>
                A Logistic Regression model was trained <b>once</b> on past student
                data and saved to disk. This app loads that saved model and gives it
                the study hours you entered (<b>{hours}</b>). The model turns the
                hours into a pass probability. If the probability is
                <b>50% or higher</b>, the predicted class is <b>PASS</b>, otherwise
                <b>FAIL</b>. Nothing is retrained when you click the button.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Bonus chart: study hours vs predicted class
        st.markdown('<div class="card"><h3>📈 Study hours vs. predicted class</h3>', unsafe_allow_html=True)
        grid = pd.DataFrame({"study_hours": np.linspace(0, 10, 100)})
        grid["pass_probability"] = model.predict_proba(grid)[:, 1]

        zones = pd.DataFrame(
            {
                "start": [0, 0.5],
                "end": [0.5, 1],
                "zone": ["FAIL zone", "PASS zone"],
            }
        )
        bands = (
            alt.Chart(zones)
            .mark_rect(opacity=0.35)
            .encode(
                y=alt.Y("start:Q", scale=alt.Scale(domain=[0, 1])),
                y2="end:Q",
                color=alt.Color(
                    "zone:N",
                    scale=alt.Scale(domain=["FAIL zone", "PASS zone"], range=["#ffc2d6", LAVENDER]),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
            )
        )
        line = (
            alt.Chart(grid)
            .mark_line(color=PINK, strokeWidth=3)
            .encode(
                x=alt.X("study_hours:Q", title="Study hours per day"),
                y=alt.Y("pass_probability:Q", title="Pass probability", scale=alt.Scale(domain=[0, 1])),
            )
        )
        me = pd.DataFrame({"study_hours": [hours], "pass_probability": [prob], "who": [name]})
        point = (
            alt.Chart(me)
            .mark_point(filled=True, size=220, color=PLUM, stroke="white", strokeWidth=2)
            .encode(
                x="study_hours:Q",
                y="pass_probability:Q",
                tooltip=["who", "study_hours", alt.Tooltip("pass_probability:Q", format=".0%")],
            )
        )
        chart = (
            (bands + line + point)
            .properties(height=300)
            .configure_view(strokeWidth=0)
            .configure(background="transparent")
            .configure_axis(labelColor=PLUM, titleColor=PLUM, gridColor="#ffd6e8")
            .configure_legend(labelColor=PLUM)
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            f"The dark dot is {name}: {hours} study hours → "
            f"{'PASS (1)' if pred else 'FAIL (0)'}."
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="footer">Made with 💖 using Streamlit &amp; scikit-learn</div>', unsafe_allow_html=True)