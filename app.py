import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the model ONCE when the server starts (no retraining needed)
model = joblib.load("student_pass_model_v2.joblib")

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Student Pass Predictor</title>
  <style>
    :root {
      --bg: #f1f5f9; --card: #ffffff; --text: #0f172a; --muted: #64748b;
      --border: #e2e8f0; --accent: #4f46e5; --accent-dark: #4338ca;
      --pass: #16a34a; --pass-bg: #dcfce7; --fail: #dc2626; --fail-bg: #fee2e2;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --muted: #94a3b8;
        --border: #334155; --accent: #818cf8; --accent-dark: #6366f1;
        --pass: #4ade80; --pass-bg: #14532d; --fail: #f87171; --fail-bg: #7f1d1d;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
      background: var(--bg); color: var(--text); padding: 16px;
    }
    .card {
      width: 100%; max-width: 440px; background: var(--card); border: 1px solid var(--border);
      border-radius: 16px; padding: 28px; box-shadow: 0 10px 30px rgba(0,0,0,.08);
    }
    h1 { margin: 0 0 4px; font-size: 1.5rem; }
    .sub { margin: 0 0 24px; color: var(--muted); font-size: .9rem; }
    .field { margin-bottom: 22px; }
    .row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    label { font-weight: 600; font-size: .9rem; }
    .num {
      width: 84px; padding: 6px 8px; border: 1px solid var(--border); border-radius: 8px;
      background: transparent; color: var(--text); font-size: .95rem; text-align: right;
    }
    .num:focus { outline: 2px solid var(--accent); border-color: transparent; }
    input[type=range] { width: 100%; accent-color: var(--accent); }
    .hint { display: flex; justify-content: space-between; color: var(--muted); font-size: .75rem; }
    button {
      width: 100%; padding: 12px; border: 0; border-radius: 10px; cursor: pointer;
      background: var(--accent); color: #fff; font-size: 1rem; font-weight: 600;
      transition: background .15s;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: .6; cursor: wait; }
    .result { display: none; margin-top: 22px; padding: 18px; border-radius: 12px; text-align: center; }
    .result.pass { background: var(--pass-bg); color: var(--pass); }
    .result.fail { background: var(--fail-bg); color: var(--fail); }
    .verdict { font-size: 1.8rem; font-weight: 800; margin: 0; }
    .detail { margin: 6px 0 0; font-size: .85rem; opacity: .9; }
    .bar { height: 8px; border-radius: 99px; background: rgba(127,127,127,.25); margin-top: 14px; overflow: hidden; }
    .fill { height: 100%; width: 0; background: currentColor; border-radius: 99px; transition: width .5s ease; }
    .prob { margin-top: 6px; font-size: .8rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Student Pass Predictor</h1>
    <p class="sub">Adjust the values and see the model's prediction.</p>

    <div class="field">
      <div class="row">
        <label for="hours">Study hours per day</label>
        <input class="num" id="hoursNum" type="number" min="0" max="10" step="0.5" value="5">
      </div>
      <input id="hours" type="range" min="0" max="10" step="0.5" value="5">
      <div class="hint"><span>0 h</span><span>10 h</span></div>
    </div>

    <div class="field">
      <div class="row">
        <label for="attendance">Attendance (%)</label>
        <input class="num" id="attNum" type="number" min="0" max="100" step="1" value="80">
      </div>
      <input id="attendance" type="range" min="0" max="100" step="1" value="80">
      <div class="hint"><span>0%</span><span>100%</span></div>
    </div>

    <button id="btn" onclick="predict()">Predict</button>

    <div id="result" class="result">
      <p id="verdict" class="verdict"></p>
      <p id="detail" class="detail"></p>
      <div id="barWrap" class="bar"><div id="fill" class="fill"></div></div>
      <div id="prob" class="prob"></div>
    </div>
  </div>

  <script>
    // Keep each slider and its number box in sync
    function link(rangeId, numId) {
      const r = document.getElementById(rangeId), n = document.getElementById(numId);
      r.addEventListener("input", () => n.value = r.value);
      n.addEventListener("input", () => { if (n.value !== "") r.value = n.value; });
    }
    link("hours", "hoursNum");
    link("attendance", "attNum");

    document.addEventListener("keydown", e => { if (e.key === "Enter") predict(); });

    async function predict() {
      const btn = document.getElementById("btn");
      const box = document.getElementById("result");
      btn.disabled = true; btn.textContent = "Predicting...";
      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            study_hours: document.getElementById("hoursNum").value,
            attendance_percentage: document.getElementById("attNum").value
          })
        });
        const data = await res.json();
        box.style.display = "block";
        const bar = document.getElementById("barWrap");
        if (!res.ok) {
          box.className = "result fail";
          document.getElementById("verdict").textContent = "Invalid input";
          document.getElementById("detail").textContent = data.error;
          bar.style.display = "none";
          document.getElementById("prob").textContent = "";
        } else {
          const passed = data.prediction === "Pass";
          box.className = "result " + (passed ? "pass" : "fail");
          document.getElementById("verdict").textContent = passed ? "PASS" : "FAIL";
          document.getElementById("detail").textContent =
            data.study_hours + " study hours, " + data.attendance_percentage + "% attendance";
          bar.style.display = "block";
          const pct = Math.round(data.pass_probability * 100);
          document.getElementById("fill").style.width = pct + "%";
          document.getElementById("prob").textContent = "Pass probability: " + pct + "%";
        }
      } catch (err) {
        box.style.display = "block";
        box.className = "result fail";
        document.getElementById("verdict").textContent = "Server error";
        document.getElementById("detail").textContent = "Is the Flask server running?";
      }
      btn.disabled = false; btn.textContent = "Predict";
    }
  </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Send a JSON body"}), 400

    # Check the fields exist
    for field in ("study_hours", "attendance_percentage"):
        if field not in body:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Check they are numbers
    try:
        study_hours = float(body["study_hours"])
        attendance = float(body["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers"}), 400

    # Validate ranges
    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or more"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    features = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["study_hours", "attendance_percentage"],
    )
    result = int(model.predict(features)[0])
    pass_probability = float(model.predict_proba(features)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if result == 1 else "Fail",
        "pass_probability": round(pass_probability, 3),
    })


if __name__ == "__main__":
    app.run(debug=True)