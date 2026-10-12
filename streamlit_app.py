import hashlib

import joblib
import pandas as pd
import requests
import streamlit as st

from evaluate_model import evaluate

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

# ---------- Students (BSIT 4 Block 2) ----------
STUDENTS = [
    "Abayle, John Paul", "Aldaya, Krisihia Mae", "Amolata, Ma. Leslie Janna",
    "Ampo, Aleah", "Aniasco, Jamel", "Aro, Kian Kenneth", "Auguis, Jerome",
    "Besan, Ashley Joy", "Cabañero, Jean Chawe", "Cagampang, John Paul",
    "Camposo, Jay Marvin", "Cerencio, Jan Mel", "Colarte, Irene",
    "Crompido, Genie Mae", "Cutamora, Rainalyn", "Dano, Vincent Rey",
    "Dologuin, Jezza", "Galgo, Shanchai", "Gandulpos, Jaciel",
    "Guibay, Khey Marie", "Guisihan, Jenny Rose", "Guzman, Irine",
    "Labajo, Bea Mae", "Lagare, Marielle", "Ludevice, Darwin",
    "Manicio, Ray Christian", "Muñasque, Janice", "Palmero, Imee",
    "Patron, Jessamie", "Payot, Michael Dave", "Pracullos, Antonett",
    "Ramos, Mary Grace", "Roscual, Alisa Mae", "Rubion, Charm",
    "Sumatra, Jayremi", "Taghoy, Cherry Ruth", "Torrenueva, Desie",
    "Tura, Ivy Jane", "Turtoga, Rhyn Carylle", "Zapanta, Michael John",
]


CLASS_NAME = "BSIT - 4 Block 2"
API_URL = "http://127.0.0.1:5000/predict"  # Flask API from app.py


# ---------- Demo accounts (passwords stored as hashes) ----------
def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


USERS = {
    "admin": {"password": hash_pw("admin123"), "name": "Administrator"},
    "teacher": {"password": hash_pw("teacher123"), "name": "Teacher"},
}


# ---------- Model ----------
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


@st.cache_data
def get_evaluation():
    return evaluate()


def predict_student(name, study_hours, attendance):
    """Ask the Flask API for a prediction; fall back to the local model if it is offline."""
    payload = {
        "student_name": name,
        "study_hours": study_hours,
        "attendance_percentage": attendance,
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=3)
        data = resp.json()
        if resp.status_code == 200:
            return {
                "prediction": data["prediction"],
                "confidence": data["confidence"],
                "source": "Flask API",
            }
        return {"error": data.get("error", "The API returned an error.")}
    except requests.exceptions.RequestException:
        model = load_model()
        features = [[study_hours, attendance]]
        pred = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        return {
            "prediction": "Pass" if pred == 1 else "Fail",
            "confidence": round(float(max(proba)) * 100, 2),
            "source": "local model (Flask API is not running)",
        }


# ---------- Session state ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = ""
    st.session_state.history = []

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp { font-family: 'Inter', sans-serif; }
    .stApp {
        background:
            radial-gradient(1000px 500px at 10% -10%, #1e1b4b 0%, transparent 60%),
            radial-gradient(800px 450px at 100% 0%, #0c4a6e 0%, transparent 55%),
            #0b1020;
    }
    .stApp, .stApp p, .stApp label, .stApp li { color: #e5e7eb; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 780px; }

    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.03);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #06b6d4 120%);
        padding: 1.8rem 2rem; border-radius: 22px; margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(99,102,241,0.35);
    }
    .hero h1 { color: #fff !important; margin: 0; font-size: 1.9rem; font-weight: 700; }
    .hero p { color: rgba(255,255,255,0.88) !important; margin: .4rem 0 0 0; }
    .pill {
        display: inline-block; background: rgba(255,255,255,0.18); color: #fff;
        padding: .25rem .8rem; border-radius: 999px; font-size: .8rem;
        font-weight: 600; letter-spacing: .04em; margin-bottom: .7rem;
        border: 1px solid rgba(255,255,255,0.25);
    }

    /* Cards */
    .card, div[data-testid="stForm"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px; padding: 1.2rem 1.3rem;
        backdrop-filter: blur(8px);
    }
    .stat {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px; padding: 1rem 1.1rem;
    }
    .stat-label { color: #9ca3af; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }
    .stat-value { color: #fff; font-size: 1.7rem; font-weight: 700; margin-top: .15rem; }

    .badge {
        display: inline-block; padding: .3rem .8rem; border-radius: 999px;
        font-weight: 600; font-size: .85rem;
    }
    .bar { background: rgba(255,255,255,0.1); border-radius: 999px; height: 10px; margin: .8rem 0 .4rem 0; overflow: hidden; }
    .bar > div { height: 100%; border-radius: 999px; }
    .muted { color: #9ca3af; font-size: .85rem; }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        border-color: rgba(255,255,255,0.12) !important;
    }

    /* Buttons */
    .stButton > button, div[data-testid="stFormSubmitButton"] > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #fff; border: 0; border-radius: 12px;
        padding: .6rem 1rem; font-weight: 600;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 8px 24px rgba(99,102,241,0.45); color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def stat(label, value):
    return f"<div class='stat'><div class='stat-label'>{label}</div><div class='stat-value'>{value}</div></div>"


def result_card(name, result, probability):
    ok = result == "Pass"
    color = "#22c55e" if ok else "#ef4444"
    icon = "✅" if ok else "⚠️"
    return f"""
    <div class="card" style="border-color:{color}66;">
        <span class="badge" style="background:{color}22;color:{color};">{icon} Predicted {result.upper()}</span>
        <h3 style="margin:.7rem 0 0 0;color:#fff;">{name}</h3>
        <div class="bar"><div style="width:{probability * 100:.0f}%;background:{color};"></div></div>
        <div class="muted">Confidence: {probability:.0%}</div>
    </div>
    """


# ---------- Login page ----------
def login_page():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            f"<div class='hero' style='text-align:center'>"
            f"<span class='pill'>{CLASS_NAME}</span>"
            "<h1>🎓 Student Pass Predictor</h1>"
            "<p>Sign in to continue</p></div>",
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", width="stretch")

        if submitted:
            user = USERS.get(username.strip().lower())
            if user and user["password"] == hash_pw(password):
                st.session_state.logged_in = True
                st.session_state.user = user["name"]
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.caption("Demo accounts: admin / admin123 · teacher / teacher123")


# ---------- Main app ----------
def main_page():
    model = load_model()
    history = st.session_state.history

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user}")
        st.caption(CLASS_NAME)
        done = len({h["Student"] for h in history})
        st.markdown(f"**Students predicted:** {done} / {len(STUDENTS)}")
        st.progress(done / len(STUDENTS))
        st.write("")
        if st.button("Log out", width="stretch"):
            st.session_state.logged_in = False
            st.session_state.user = ""
            st.session_state.history = []
            st.rerun()

    st.markdown(
        f"<div class='hero'><span class='pill'>{CLASS_NAME}</span>"
        "<h1>🎓 Student Pass Predictor</h1>"
        "<p>Select a student, enter study hours and attendance, and get an instant prediction.</p></div>",
        unsafe_allow_html=True,
    )

    tab_predict, tab_history, tab_eval = st.tabs(["📊 Predict", "🗂️ History", "📈 Evaluation"])

    # ----- Predict tab -----
    with tab_predict:
        with st.form("predict_form"):
            name = st.selectbox(
                "Student",
                STUDENTS,
                index=None,
                placeholder="Search or select a student...",
            )
            c1, c2 = st.columns(2)
            study_hours = c1.number_input(
                "Study hours per day", min_value=0.0, max_value=24.0, value=4.0, step=0.5
            )
            attendance = c2.slider("Attendance (%)", 0, 100, 80)
            submitted = st.form_submit_button("Predict", width="stretch")

        if submitted:
            if not name:
                st.warning("Please select a student from the list.")
            else:
                out = predict_student(name, study_hours, attendance)
                if "error" in out:
                    st.error(out["error"])
                    st.stop()
                result = out["prediction"]
                probability = out["confidence"] / 100
                source = out["source"]

                # Keep only the latest prediction per student
                st.session_state.history = [h for h in history if h["Student"] != name] + [
                    {
                        "Student": name,
                        "Study hours": study_hours,
                        "Attendance (%)": attendance,
                        "Result": result,
                        "Confidence (%)": out["confidence"],
                    }
                ]

                st.write("")
                st.markdown(result_card(name, result, probability), unsafe_allow_html=True)
                st.caption(f"Predicted by: {source}")

                st.write("")
                m1, m2, m3 = st.columns(3)
                m1.markdown(stat("Study hours", f"{study_hours:g}"), unsafe_allow_html=True)
                m2.markdown(stat("Attendance", f"{attendance}%"), unsafe_allow_html=True)
                m3.markdown(stat("Confidence", f"{probability:.0%}"), unsafe_allow_html=True)

                st.write("")
                with st.expander("How was this predicted?"):
                    st.write(
                        f"The app sent {study_hours:g} study hours and {attendance}% attendance "
                        "to the Flask API (`app.py`), which loads a logistic regression model "
                        "saved earlier in `student_pass_model.joblib`. The model was trained once, "
                        "so no retraining happens here. It calculates the probability of passing "
                        "from those two values. If that is 50% or higher the result is Pass, "
                        "otherwise Fail, and the confidence is the probability of the predicted class."
                    )

                st.subheader("Study hours vs. predicted class")
                hours_range = [h / 2 for h in range(0, 33)]
                probs = model.predict_proba([[h, attendance] for h in hours_range])[:, 1]
                chart_data = pd.DataFrame(
                    {"Pass probability": probs},
                    index=pd.Index(hours_range, name="Study hours"),
                )
                st.line_chart(chart_data)
                st.caption(f"At {attendance}% attendance. Above 0.5 the predicted class is Pass.")

    # ----- History tab -----
    with tab_history:
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)

            passed = int((df["Result"] == "Pass").sum())
            failed = int((df["Result"] == "Fail").sum())
            h1, h2, h3 = st.columns(3)
            h1.markdown(stat("Predictions", len(df)), unsafe_allow_html=True)
            h2.markdown(stat("Pass", passed), unsafe_allow_html=True)
            h3.markdown(stat("Fail", failed), unsafe_allow_html=True)

            st.write("")
            st.dataframe(df, width="stretch", hide_index=True)

            counts = df["Result"].value_counts().reindex(["Pass", "Fail"], fill_value=0)
            st.bar_chart(counts)

            st.download_button(
                "⬇️ Download as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        else:
            st.info("No predictions yet. Make one in the Predict tab.")

    # ----- Evaluation tab -----
    with tab_eval:
        r = get_evaluation()
        st.subheader("Model evaluation")
        st.caption("3-fold cross-validation on the 10 training records.")

        e1, e2, e3, e4 = st.columns(4)
        e1.markdown(stat("Accuracy", f"{r['accuracy']:.0%}"), unsafe_allow_html=True)
        e2.markdown(stat("Precision", f"{r['precision']:.0%}"), unsafe_allow_html=True)
        e3.markdown(stat("Recall", f"{r['recall']:.0%}"), unsafe_allow_html=True)
        e4.markdown(stat("F1-Score", f"{r['f1']:.0%}"), unsafe_allow_html=True)

        st.write("")
        st.markdown("**Confusion matrix**")
        cm_df = pd.DataFrame(
            [[r["TP"], r["FN"]], [r["FP"], r["TN"]]],
            index=["Actual Pass", "Actual Fail"],
            columns=["Predicted Pass", "Predicted Fail"],
        )
        st.dataframe(cm_df, width="stretch")

        with st.expander("What do these numbers mean?"):
            st.write(
                f"- **TP = {r['TP']}**: passing students correctly predicted as Pass.\n"
                f"- **TN = {r['TN']}**: failing students correctly predicted as Fail.\n"
                f"- **FP = {r['FP']}**: failing students wrongly predicted as Pass.\n"
                f"- **FN = {r['FN']}**: passing students wrongly predicted as Fail.\n\n"
                "**Accuracy** is the share of all predictions that were correct. "
                "**Precision** is how often a Pass prediction is right. "
                "**Recall** is how many of the real passers the model found. "
                "**F1-Score** balances precision and recall."
            )


# ---------- Router ----------
if st.session_state.logged_in:
    main_page()
else:
    login_page()