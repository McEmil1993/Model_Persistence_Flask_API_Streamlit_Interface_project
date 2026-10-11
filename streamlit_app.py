"""Exercise 3 - Streamlit interface (uses the study-hours model)."""
import time

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon=None, layout="centered")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 820px;}
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        padding: 2rem 1.75rem; border-radius: 18px; color: #fff; margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, .25);
    }
    .hero h1 {margin: 0; font-size: 2.1rem; color: #fff; padding: 0;}
    .hero p {margin: .4rem 0 0; opacity: .92; font-size: 1.02rem;}
    .result {
        border-radius: 16px; padding: 1.4rem 1.5rem; margin: 1rem 0 .5rem;
        border: 1px solid;
    }
    .result.pass {background: rgba(16,185,129,.12); border-color: rgba(16,185,129,.5);}
    .result.fail {background: rgba(239,68,68,.12);  border-color: rgba(239,68,68,.5);}
    .result .title {font-size: 1.35rem; font-weight: 700; margin: 0;}
    .result .sub {margin: .15rem 0 0; opacity: .8;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.2);
        padding: .8rem 1rem; border-radius: 14px;
    }
    /* ---- Predict button ---- */
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 3.2rem;
        border: none;
        border-radius: 14px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        background-size: 160% 160%;
        background-position: 0% 50%;
        color: #fff;
        box-shadow: 0 6px 18px rgba(124, 58, 237, .35);
        transition: transform .15s ease, box-shadow .15s ease, background-position .4s ease;
    }
    div[data-testid="stFormSubmitButton"] button p {
        color: #fff; font-size: 1.1rem; font-weight: 700; letter-spacing: .04em;
        text-transform: uppercase;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background-position: 100% 50%;
        transform: translateY(-2px);
        box-shadow: 0 10px 26px rgba(124, 58, 237, .5);
        color: #fff;
    }
    div[data-testid="stFormSubmitButton"] button:active {
        transform: translateY(0) scale(.98);
        box-shadow: 0 3px 10px rgba(124, 58, 237, .35);
    }
    div[data-testid="stFormSubmitButton"] button:focus-visible {
        outline: 3px solid rgba(124, 58, 237, .45); outline-offset: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "This demo loads a **saved Logistic Regression model** "
        "(`student_pass_model.joblib`) and predicts whether a student will pass "
        "based on study hours."
    )
    st.markdown("**Model details**")
    st.markdown(
        "- Input: study hours\n"
        "- Output: Pass / Fail\n"
        "- Trained on 0.5 – 10 hours\n"
        "- No retraining in this app"
    )
    st.caption("IT Elective 3 · Model Persistence & Deployment")

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>Student Pass Predictor</h1>
        <p>Enter a student's name and study hours to see if they're likely to pass.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Input form ----------
with st.form("predict_form"):
    name = st.text_input("Student name", placeholder="e.g. Maria Santos")
    hours = st.slider("Study hours", min_value=0.0, max_value=24.0,
                      value=3.0, step=0.5, help="Hours the student studied.")
    submitted = st.form_submit_button("Predict", type="primary")

# ---------- Prediction ----------
if submitted:
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        with st.status("Prediction in progress...", expanded=True) as status:
            st.write("Loading the saved model...")
            time.sleep(0.6)
            st.write(f"Reading {hours:g} study hours for {name.strip()}...")
            time.sleep(0.6)
            X = pd.DataFrame({"study_hours": [hours]})
            st.write("Calculating the pass probability...")
            time.sleep(0.6)
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])
            label = "Pass" if pred == 1 else "Fail"
            status.update(label="Prediction complete", state="complete", expanded=False)

        # Result message
        css = "pass" if pred == 1 else "fail"
        verb = "is predicted to PASS" if pred == 1 else "is predicted to FAIL"
        st.markdown(
            f"""
            <div class="result {css}">
                <div>
                    <p class="title">{name.strip()} {verb}</p>
                    <p class="sub">Pass probability: {prob:.0%}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if hours > 10:
            st.info("The model was trained on up to 10 study hours, so results "
                    "above that are extrapolated.")

        # Key numbers
        c1, c2, c3 = st.columns(3)
        c1.metric("Study hours", f"{hours:g} h")
        c2.metric("Pass probability", f"{prob:.0%}")
        c3.metric("Predicted class", label)
        st.progress(prob, text=f"Confidence of passing: {prob:.0%} (threshold 50%)")

        # Tabs: explanation + chart
        tab1, tab2 = st.tabs(["How it was predicted", "Hours vs. class"])

        with tab1:
            st.markdown(
                f"""
                1. **Load** – the app loads a Logistic Regression model that was
                   trained earlier and saved with `joblib`.
                2. **Input** – your study hours (**{hours:g} h**) are given to the model.
                3. **Probability** – the model converts the hours into a pass
                   probability (**{prob:.0%}**).
                4. **Class** – if the probability is **50% or higher**, the class is
                   *Pass*; otherwise it is *Fail*.

                The model is **not retrained** here. It only applies what it
                already learned.
                """
            )

        with tab2:
            grid = pd.DataFrame({"study_hours": [i / 4 for i in range(0, 97)]})
            grid["pass_probability"] = model.predict_proba(grid)[:, 1]
            point = pd.DataFrame({"study_hours": [hours], "pass_probability": [prob],
                                  "class": [label]})
            thresh = pd.DataFrame({"y": [0.5]})

            line = alt.Chart(grid).mark_line(color="#7c3aed", strokeWidth=3).encode(
                x=alt.X("study_hours:Q", title="Study hours"),
                y=alt.Y("pass_probability:Q", title="Pass probability",
                        scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            )
            rule = alt.Chart(thresh).mark_rule(strokeDash=[6, 4], color="gray").encode(y="y:Q")
            dot = alt.Chart(point).mark_point(size=260, filled=True, opacity=1).encode(
                x="study_hours:Q", y="pass_probability:Q",
                color=alt.Color("class:N", legend=alt.Legend(title="Predicted class"),
                                scale=alt.Scale(domain=["Pass", "Fail"],
                                                range=["#10b981", "#ef4444"])),
                tooltip=["study_hours", alt.Tooltip("pass_probability:Q", format=".0%"), "class"],
            )
            st.altair_chart((line + rule + dot).properties(height=330),
                            width="stretch")
            st.caption(f"{name.strip()} studies {hours:g} h → predicted class: **{label}**. "
                       "The dashed line is the 50% decision threshold.")
else:
    st.info("Fill in the form and press **Predict** to see the result.")