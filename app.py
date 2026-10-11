from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PassPredict | Student Performance Predictor</title>
<style>
  :root {
    --primary: #4f46e5;
    --primary-dark: #4338ca;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --bg: #f1f5f9;
    --green: #16a34a;
    --red: #dc2626;
    --amber: #d97706;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* ---------- Navbar ---------- */
  .nav {
    background: #fff;
    border-bottom: 1px solid var(--border);
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 20px; }
  .logo {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, #4f46e5, #06b6d4);
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .badge {
    font-size: 12px; font-weight: 600; color: var(--primary);
    background: #eef2ff; padding: 5px 12px; border-radius: 999px;
  }

  /* ---------- Layout ---------- */
  .wrap {
    max-width: 1050px;
    margin: 40px auto;
    padding: 0 20px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
  }
  @media (max-width: 820px) { .wrap { grid-template-columns: 1fr; margin: 24px auto; } }

  .hero {
    background: linear-gradient(145deg, #4f46e5 0%, #7c3aed 55%, #06b6d4 120%);
    color: #fff;
    border-radius: 20px;
    padding: 40px 34px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 20px 40px rgba(79, 70, 229, 0.25);
  }
  .hero h1 { font-size: 32px; line-height: 1.2; margin-bottom: 14px; }
  .hero p { opacity: 0.9; line-height: 1.6; font-size: 15px; }
  .features { list-style: none; margin-top: 26px; }
  .features li {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 14px; font-size: 14px;
  }
  .features span.ic {
    width: 32px; height: 32px; border-radius: 9px;
    background: rgba(255,255,255,0.18);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
  }
  .foot { margin-top: 26px; font-size: 12px; opacity: 0.75; }

  /* ---------- Card ---------- */
  .card {
    background: #fff;
    border-radius: 20px;
    padding: 34px;
    border: 1px solid var(--border);
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
  }
  .card h2 { font-size: 22px; margin-bottom: 4px; }
  .card .sub { color: var(--muted); font-size: 14px; margin-bottom: 26px; }

  .field { margin-bottom: 24px; }
  .field-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .field-head label { font-size: 14px; font-weight: 600; }
  .value-pill {
    background: #eef2ff; color: var(--primary);
    font-weight: 700; font-size: 14px;
    padding: 4px 12px; border-radius: 999px; min-width: 64px; text-align: center;
  }
  input[type=range] {
    width: 100%; height: 6px; border-radius: 6px;
    background: var(--border); outline: none; -webkit-appearance: none;
    appearance: none; cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 22px; height: 22px; border-radius: 50%;
    background: var(--primary); border: 4px solid #fff;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.5);
  }
  input[type=range]::-moz-range-thumb {
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--primary); border: 4px solid #fff;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.5);
  }
  .scale { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 6px; }

  button.predict {
    width: 100%; padding: 14px; border: none; border-radius: 12px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #fff; font-size: 16px; font-weight: 600; cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.35);
    display: flex; align-items: center; justify-content: center; gap: 10px;
  }
  button.predict:hover { transform: translateY(-2px); }
  button.predict:disabled { opacity: 0.7; cursor: wait; transform: none; }
  .spinner {
    width: 18px; height: 18px; border: 3px solid rgba(255,255,255,0.4);
    border-top-color: #fff; border-radius: 50%;
    animation: spin 0.7s linear infinite; display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ---------- Result ---------- */
  #result { margin-top: 22px; display: none; }
  .res {
    border-radius: 14px; padding: 20px; display: flex; gap: 16px; align-items: center;
    animation: pop 0.35s ease;
  }
  @keyframes pop { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
  .res .icon {
    width: 52px; height: 52px; border-radius: 50%; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; font-weight: 700; flex-shrink: 0;
  }
  .res .title { font-size: 20px; font-weight: 700; }
  .res .desc { font-size: 13px; margin-top: 3px; }
  .res.pass { background: #f0fdf4; border: 1px solid #bbf7d0; }
  .res.pass .icon { background: var(--green); }
  .res.pass .title { color: #166534; }
  .res.pass .desc { color: #15803d; }
  .res.fail { background: #fef2f2; border: 1px solid #fecaca; }
  .res.fail .icon { background: var(--red); }
  .res.fail .title { color: #991b1b; }
  .res.fail .desc { color: #b91c1c; }
  .res.error { background: #fffbeb; border: 1px solid #fde68a; }
  .res.error .icon { background: var(--amber); }
  .res.error .title { color: #92400e; font-size: 16px; }
  .res.error .desc { color: #b45309; }

  /* ---------- History ---------- */
  #history-box { margin-top: 24px; display: none; }
  #history-box h3 { font-size: 14px; color: var(--muted); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: var(--muted); font-weight: 600; padding: 8px 6px; border-bottom: 1px solid var(--border); }
  td { padding: 9px 6px; border-bottom: 1px solid var(--border); }
  .tag { padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .tag.pass { background: #dcfce7; color: #166534; }
  .tag.fail { background: #fee2e2; color: #991b1b; }
</style>
</head>
<body>

<div class="nav">
  <div class="brand"><div class="logo">&#127891;</div> PassPredict</div>
  <div class="badge">Machine Learning &bull; Logistic Regression</div>
</div>

<div class="wrap">

  <div class="hero">
    <div>
      <h1>Will the student pass?</h1>
      <p>A machine learning model trained on study hours and class attendance
         predicts whether a student is likely to pass or fail.</p>
      <ul class="features">
        <li><span class="ic">&#9201;</span> Uses study hours as a factor</li>
        <li><span class="ic">&#128197;</span> Uses attendance percentage as a factor</li>
        <li><span class="ic">&#9989;</span> Validated inputs (attendance 0 to 100)</li>
        <li><span class="ic">&#9889;</span> Instant prediction through a Flask API</li>
      </ul>
    </div>
    <div class="foot">Built with Flask, scikit-learn and joblib</div>
  </div>

  <div class="card">
    <h2>Check a student</h2>
    <div class="sub">Adjust the sliders, then press Predict.</div>

    <div class="field">
      <div class="field-head">
        <label for="hours">Study hours</label>
        <div class="value-pill" id="hours-val">6 hrs</div>
      </div>
      <input type="range" id="hours" min="0" max="12" step="0.5" value="6">
      <div class="scale"><span>0</span><span>6</span><span>12</span></div>
    </div>

    <div class="field">
      <div class="field-head">
        <label for="attendance">Attendance percentage</label>
        <div class="value-pill" id="attendance-val">80%</div>
      </div>
      <input type="range" id="attendance" min="0" max="100" step="1" value="80">
      <div class="scale"><span>0%</span><span>50%</span><span>100%</span></div>
    </div>

    <button class="predict" id="btn" onclick="predict()">
      <div class="spinner" id="spin"></div>
      <span id="btn-text">Predict Result</span>
    </button>

    <div id="result"></div>

    <div id="history-box">
      <h3>Recent predictions</h3>
      <table>
        <thead><tr><th>Study hours</th><th>Attendance</th><th>Result</th></tr></thead>
        <tbody id="history"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
const hours = document.getElementById("hours");
const attendance = document.getElementById("attendance");

hours.addEventListener("input", function () {
  document.getElementById("hours-val").textContent = hours.value + " hrs";
});
attendance.addEventListener("input", function () {
  document.getElementById("attendance-val").textContent = attendance.value + "%";
});

function showResult(kind, icon, title, desc) {
  const box = document.getElementById("result");
  box.style.display = "block";
  box.innerHTML =
    '<div class="res ' + kind + '">' +
      '<div class="icon">' + icon + '</div>' +
      '<div><div class="title">' + title + '</div>' +
      '<div class="desc">' + desc + '</div></div>' +
    '</div>';
}

function addHistory(h, a, prediction) {
  document.getElementById("history-box").style.display = "block";
  const tbody = document.getElementById("history");
  const row = document.createElement("tr");
  const cls = prediction === "Pass" ? "pass" : "fail";
  row.innerHTML =
    "<td>" + h + " hrs</td><td>" + a + "%</td>" +
    '<td><span class="tag ' + cls + '">' + prediction + "</span></td>";
  tbody.insertBefore(row, tbody.firstChild);
  while (tbody.children.length > 5) {
    tbody.removeChild(tbody.lastChild);
  }
}

async function predict() {
  const btn = document.getElementById("btn");
  const spin = document.getElementById("spin");
  const text = document.getElementById("btn-text");

  btn.disabled = true;
  spin.style.display = "block";
  text.textContent = "Predicting...";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        study_hours: Number(hours.value),
        attendance_percentage: Number(attendance.value)
      })
    });
    const data = await res.json();

    if (!res.ok) {
      showResult("error", "!", "Invalid input", data.error);
    } else if (data.prediction === "Pass") {
      showResult("pass", "&#10003;", "Likely to PASS",
        data.study_hours + " study hours with " + data.attendance_percentage + "% attendance");
      addHistory(data.study_hours, data.attendance_percentage, data.prediction);
    } else {
      showResult("fail", "&#10007;", "Likely to FAIL",
        data.study_hours + " study hours with " + data.attendance_percentage + "% attendance");
      addHistory(data.study_hours, data.attendance_percentage, data.prediction);
    }
  } catch (e) {
    showResult("error", "!", "Server not reachable", "Make sure the Flask app is running.");
  }

  btn.disabled = false;
  spin.style.display = "none";
  text.textContent = "Predict Result";
}
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return PAGE


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "study_hours and attendance_percentage are required"}), 400

    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers"}), 400

    if hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400

    # Validation: attendance must be between 0 and 100
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    result = int(model.predict([[hours, attendance]])[0])

    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if result == 1 else "Fail"
    })


if __name__ == "__main__":
    app.run(debug=True)