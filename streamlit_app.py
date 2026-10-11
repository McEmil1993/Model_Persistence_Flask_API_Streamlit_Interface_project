import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Pass Predictor", layout="centered")

st.markdown(
    """
    <style>
        .block-container { max-width: 760px; padding-top: 2rem; }
        .hero {
            background: linear-gradient(135deg, #064e3b 0%, #10b981 100%);
            color: #ffffff;
            padding: 1.5rem 1.75rem;
            border-radius: 14px;
            margin-bottom: 1.25rem;
        }
        .hero h1 { color: #ffffff; margin: 0; font-size: 1.9rem; }
        .hero p { color: #d1fae5; margin: 0.35rem 0 0 0; }
    </style>
    <div class="hero">
        <h1>Student Pass Predictor</h1>
        <p>Predict whether a student will pass, based on study hours.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = joblib.load("student_pass_model.joblib")

# This page only uses study hours. Stop if the saved model also expects attendance.
if "attendance_percentage" in model.feature_names_in_:
    st.error(
        "The saved model needs attendance. Set USE_ATTENDANCE = False in "
        "train_model.py, run it again, then restart Streamlit."
    )
    st.stop()

# Training data from train_model.py, used to show past students on the graph.
TRAIN_HOURS = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7])
TRAIN_PASSED = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])

PASS_COLOR = "#16a34a"
FAIL_COLOR = "#dc2626"
ACCENT_COLOR = "#059669"
GRAY_COLOR = "#6b7280"

# ----- Inputs -----
with st.container(border=True):
    name = st.text_input("Student name")
    study_hours = st.number_input(
        "Study hours per day", min_value=0.0, max_value=24.0, value=0.0, step=0.5
    )
    predict_clicked = st.button("Predict", type="primary")

# ----- Results -----
if not predict_clicked:
    st.info("Enter the student's name and study hours, then click Predict.")
elif not name.strip():
    st.warning("Please enter the student's name.")
else:
    sample = pd.DataFrame([[study_hours]], columns=["study_hours"])
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])
    label = "Pass" if prediction == 1 else "Fail"

    if prediction == 1:
        st.success(f"**{name} is predicted to PASS.**")
    else:
        st.error(f"**{name} is predicted to FAIL.**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Study hours", f"{study_hours:g} h")
    col2.metric("Pass probability", f"{probability:.0%}")
    col3.metric("Predicted class", label)

    st.progress(probability, text=f"Pass probability: {probability:.1%}")

    # ----- How the prediction was generated -----
    weight = model.coef_[0][0]
    intercept = model.intercept_[0]
    score = intercept + weight * study_hours

    with st.container(border=True):
        st.subheader("How this was predicted")
        st.markdown(
            f"""
The model is a **logistic regression** trained on past student data.

1. **Score** = intercept + weight × study hours
   = {intercept:.3f} + {weight:.3f} × {study_hours:g} = **{score:.3f}**
2. **Probability** = 1 ÷ (1 + e^(−score)) = **{probability:.1%}**
3. If the probability is **50% or higher**, the model predicts **Pass**. Otherwise it predicts **Fail**.
"""
        )

    # ----- Graph: study hours vs. pass probability -----
    hours = np.linspace(0, 24, 200)
    curve = model.predict_proba(pd.DataFrame({"study_hours": hours}))[:, 1]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor("white")

    # Shade the pass and fail regions around the 50% threshold.
    ax.fill_between(hours, curve, 0.5, where=curve >= 0.5,
                    color=PASS_COLOR, alpha=0.10, interpolate=True)
    ax.fill_between(hours, curve, 0.5, where=curve < 0.5,
                    color=FAIL_COLOR, alpha=0.10, interpolate=True)

    # Model's probability curve.
    ax.plot(hours, curve, color=ACCENT_COLOR, linewidth=2.8,
            label="Model pass probability")

    # 50% decision threshold and its label.
    ax.axhline(0.5, linestyle="--", color=GRAY_COLOR, linewidth=1.2)
    ax.text(23.8, 0.51, "50% threshold", ha="right", va="bottom",
            fontsize=8, color=GRAY_COLOR)

    # Past students from the training data.
    passed_mask = TRAIN_PASSED == 1
    ax.scatter(TRAIN_HOURS[passed_mask], TRAIN_PASSED[passed_mask],
               color=PASS_COLOR, s=38, alpha=0.6, edgecolors="white",
               linewidths=0.6, label="Past students who passed", zorder=2)
    ax.scatter(TRAIN_HOURS[~passed_mask], TRAIN_PASSED[~passed_mask],
               color=FAIL_COLOR, s=38, alpha=0.6, edgecolors="white",
               linewidths=0.6, label="Past students who failed", zorder=2)

    # The current student's point, with a label.
    point_color = PASS_COLOR if prediction == 1 else FAIL_COLOR
    ax.scatter([study_hours], [probability], color=point_color, s=180, zorder=4,
               edgecolors="white", linewidths=2, label=f"{name}: {label}")
    ax.annotate(f"{probability:.0%}", xy=(study_hours, probability),
                xytext=(10, 12), textcoords="offset points",
                fontsize=10, fontweight="bold", color=point_color)

    ax.set_xlabel("Study hours per day", fontsize=10)
    ax.set_ylabel("Pass probability", fontsize=10)
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xticks(range(0, 25, 4))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    # Legend below the graph, in two columns.
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()

    with st.container(border=True):
        st.subheader("Study hours vs. predicted class")
        st.pyplot(fig)
        st.caption(
            f"{name} studies {study_hours:g} h and is predicted to {label.upper()}. "
            "Green shading means Pass, red shading means Fail. "
            "Dots are past students from the training data."
        )