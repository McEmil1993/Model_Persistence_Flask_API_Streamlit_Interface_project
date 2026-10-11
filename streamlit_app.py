import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # prevents the memory error on this laptop

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="wide")

# ---------- Small style touch ----------
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(120, 140, 180, 0.12);
        border: 1px solid rgba(120, 140, 180, 0.35);
        border-radius: 12px;
        padding: 14px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Models ----------
MODELS = {
    "Model 1: Study hours only": {"file": "student_pass_model.joblib", "uses_attendance": False},
    "Model 2: Study hours + Attendance": {"file": "student_pass_model_v2.joblib", "uses_attendance": True},
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def build_features(hours, attendance, uses_attendance):
    hours = np.atleast_1d(hours).astype(float)
    if uses_attendance:
        return pd.DataFrame({
            "study_hours": hours,
            "attendance_percentage": np.full_like(hours, float(attendance)),
        })
    return pd.DataFrame({"study_hours": hours})


def hours_needed(model, attendance, uses_attendance):
    """Smallest study hours (0 to 12) where pass probability reaches 50%."""
    grid = np.arange(0, 12.01, 0.25)
    probs = model.predict_proba(build_features(grid, attendance, uses_attendance))[:, 1]
    reached = np.where(probs >= 0.5)[0]
    return float(grid[reached[0]]) if len(reached) else None


# ---------- Session state ----------
if "last" not in st.session_state:
    st.session_state["last"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []


def clear_history():
    st.session_state["history"] = []
    st.session_state["last"] = None


# ---------- Sidebar (inputs) ----------
st.sidebar.title("🎓 Inputs")
model_choice = st.sidebar.selectbox("Prediction model", list(MODELS.keys()))
cfg = MODELS[model_choice]

name = st.sidebar.text_input("Student name", placeholder="e.g. Maria")
hours = st.sidebar.slider("Study hours", min_value=0.0, max_value=12.0, value=3.0, step=0.5)
if cfg["uses_attendance"]:
    attendance = st.sidebar.slider("Attendance (%)", min_value=0, max_value=100, value=80, step=1)
else:
    attendance = None
    st.sidebar.caption("This model uses study hours only.")

predict_clicked = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)
st.sidebar.button("🗑️ Clear history", on_click=clear_history, use_container_width=True)

# ---------- Load model ----------
try:
    model = load_model(cfg["file"])
except FileNotFoundError:
    st.error(f"Model file '{cfg['file']}' was not found. Run the matching training script first.")
    st.stop()

# ---------- Predict ----------
if predict_clicked:
    if not name.strip():
        st.sidebar.warning("Please enter the student's name.")
    else:
        feats = build_features(hours, attendance, cfg["uses_attendance"])
        pred = int(model.predict(feats)[0])
        prob = float(model.predict_proba(feats)[0][1])
        result = {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Student": name.strip(),
            "Model": model_choice,
            "Study hours": hours,
            "Attendance (%)": attendance if attendance is not None else "-",
            "Prediction": "Pass" if pred == 1 else "Fail",
            "Pass probability": f"{prob:.0%}",
        }
        st.session_state["last"] = {
            "name": name.strip(), "hours": hours, "attendance": attendance,
            "pred": pred, "prob": prob, "model_choice": model_choice,
        }
        st.session_state["history"].append(result)

# ---------- Main area ----------
st.title("🎓 Student Pass Predictor")
st.caption("Enter the details in the sidebar, then click Predict.")

last = st.session_state["last"]

if last is None:
    st.info("👈 Fill in the student's details in the sidebar and click **Predict** to see the result.")
else:
    # use the same model/settings that produced this result
    last_cfg = MODELS[last["model_choice"]]
    last_model = load_model(last_cfg["file"])
    needed = hours_needed(last_model, last["attendance"], last_cfg["uses_attendance"])

    # Result message
    if last["pred"] == 1:
        st.success(f"✅ {last['name']} is predicted to PASS (pass probability: {last['prob']:.0%}).")
    else:
        st.error(f"❌ {last['name']} is predicted to FAIL (pass probability: {last['prob']:.0%}).")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction", "Pass" if last["pred"] == 1 else "Fail")
    c2.metric("Pass probability", f"{last['prob']:.0%}")
    c3.metric("Study hours", f"{last['hours']}")
    if needed is None:
        c4.metric("Hours to reach 50%", "Above 12")
    else:
        c4.metric("Hours to reach 50%", f"{needed}")

    st.progress(min(max(last["prob"], 0.0), 1.0), text="Pass probability")

    if last["pred"] == 0 and needed is not None:
        gap = needed - last["hours"]
        if gap > 0:
            st.info(f"💡 About {gap:g} more study hour(s) would bring the pass probability to 50%.")

    tab_chart, tab_explain, tab_history = st.tabs(["📊 Chart", "🧠 How it works", "🕑 History"])

    # ----- Chart tab -----
    with tab_chart:
        st.subheader("Study hours vs. predicted class")
        grid = np.arange(0, 12.01, 0.5)
        grid_feats = build_features(grid, last["attendance"], last_cfg["uses_attendance"])
        chart_df = pd.DataFrame({
            "Study hours": grid,
            "Predicted class (1 = Pass, 0 = Fail)": last_model.predict(grid_feats),
            "Pass probability": last_model.predict_proba(grid_feats)[:, 1],
        }).set_index("Study hours")
        st.line_chart(chart_df)
        extra = f" at {last['attendance']}% attendance" if last_cfg["uses_attendance"] else ""
        st.caption(f"{last['name']} entered {last['hours']} study hours{extra}. Compare this value on the x-axis.")

    # ----- Explanation tab -----
    with tab_explain:
        st.subheader("How was this prediction generated?")
        used = "study hours and attendance percentage" if last_cfg["uses_attendance"] else "study hours"
        st.write(
            f"The app loaded a saved Logistic Regression model ({last_cfg['file']}) that was already "
            f"trained on past student data. It used {used} as input and computed a pass probability "
            f"of {last['prob']:.0%}. If the probability is 50% or higher, the predicted class is Pass; "
            f"otherwise, Fail. The model was not retrained."
        )
        coef_df = pd.DataFrame({
            "Feature": list(last_model.feature_names_in_),
            "Coefficient": [round(float(c), 3) for c in last_model.coef_[0]],
        })
        st.write("What the model learned (a positive coefficient raises the chance of passing):")
        st.table(coef_df)

    # ----- History tab -----
    with tab_history:
        st.subheader("Prediction history")
        if st.session_state["history"]:
            hist_df = pd.DataFrame(st.session_state["history"])
            st.table(hist_df)
            st.download_button(
                "⬇️ Download history (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
        else:
            st.write("No predictions yet.")