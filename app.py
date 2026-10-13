
from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)

# Load your trained machine-learning model
model = joblib.load("student_pass_model.joblib")

# Web interface
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Pass Prediction System</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f1f5f9;
            color: #1e293b;
        }

        header {
            background: #142d4e;
            color: white;
            padding: 22px 8%;
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .logo {
            font-size: 30px;
        }

        header h2 {
            margin: 0 0 5px;
            font-size: 21px;
        }

        header p {
            margin: 0;
            font-size: 12px;
            color: #cbd5e1;
        }

        main {
            max-width: 850px;
            margin: 45px auto;
            padding: 0 20px;
        }

        .intro {
            margin-bottom: 25px;
        }

        .intro h1 {
            font-size: 30px;
            margin-bottom: 10px;
        }

        .intro p {
            color: #64748b;
            line-height: 1.6;
        }

        .card {
            background: white;
            padding: 30px;
            border-radius: 18px;
            box-shadow: 0 8px 30px #0f172a0c;
        }

        .status {
            display: inline-block;
            background: #dcfce7;
            color: #166534;
            padding: 7px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 24px;
        }

        label {
            display: block;
            font-weight: bold;
            margin: 18px 0 8px;
        }

        input {
            width: 100%;
            padding: 13px;
            border: 1px solid #cbd5e1;
            border-radius: 9px;
            font-size: 16px;
        }

        input:focus {
            outline: 2px solid #93c5fd;
            border-color: #2563eb;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 15px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 9px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        button:disabled {
            opacity: 0.6;
            cursor: wait;
        }

        #result {
            display: none;
            margin-top: 24px;
            padding: 20px;
            border-radius: 12px;
        }

        #result h2 {
            margin: 0 0 10px;
        }

        #result p {
            margin-bottom: 0;
        }

        .pass {
            display: block !important;
            background: #dcfce7;
            color: #166534;
        }

        .fail {
            display: block !important;
            background: #fee2e2;
            color: #991b1b;
        }

        .error {
            display: block !important;
            background: #fff7ed;
            color: #9a3412;
        }

        .note {
            color: #64748b;
            font-size: 13px;
            line-height: 1.6;
            margin-top: 22px;
        }

        footer {
            text-align: center;
            color: #64748b;
            font-size: 13px;
            padding: 25px;
        }

        @media (max-width: 600px) {
            main {
                margin: 25px auto;
            }

            .card {
                padding: 22px;
            }

            .intro h1 {
                font-size: 25px;
            }
        }
    </style>
</head>

<body>
    <header>
        <div class="logo">🎓</div>
        <div>
            <h2>Student Success</h2>
            <p>PASS PREDICTION SYSTEM</p>
        </div>
    </header>

    <main>
        <div class="intro">
            <h1>Student Pass Predictor</h1>
            <p>
                Enter the student's study hours and attendance
                to generate a machine-learning prediction.
            </p>
        </div>

        <section class="card">
            <span class="status">● System Ready</span>

            <form id="predictionForm">
                <label for="study_hours">Study Hours</label>
                <input
                    type="number"
                    id="study_hours"
                    min="0"
                    step="any"
                    placeholder="Example: 5"
                    required
                >

                <label for="attendance">Attendance Percentage (%)</label>
                <input
                    type="number"
                    id="attendance"
                    min="0"
                    max="100"
                    step="any"
                    placeholder="Example: 90"
                    required
                >

                <button id="predictButton" type="submit">
                    Predict Student Result
                </button>
            </form>

            <div id="result" role="status" aria-live="polite"></div>

            <p class="note">
                Note: The current trained model uses study hours
                only. Attendance is validated but does not yet
                influence the prediction.
            </p>
        </section>
    </main>

    <footer>
        Powered by Python, Flask, Joblib and scikit-learn
    </footer>

    <script>
        const form = document.getElementById("predictionForm");
        const result = document.getElementById("result");
        const button = document.getElementById("predictButton");

        form.addEventListener("submit", async function(event) {
            event.preventDefault();

            const studyHours = Number(
                document.getElementById("study_hours").value
            );

            const attendance = Number(
                document.getElementById("attendance").value
            );

            result.className = "";
            result.style.display = "none";

            button.disabled = true;
            button.textContent = "Predicting...";

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        study_hours: studyHours,
                        attendance_percentage: attendance
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    result.className = "error";
                    result.innerHTML =
                        "<h2>Invalid Input</h2><p></p>";
                    result.querySelector("p").textContent =
                        data.error || "Something went wrong.";
                } else {
                    const isPass = data.prediction === "Pass";

                    result.className = isPass ? "pass" : "fail";
                    result.innerHTML = `
                        <h2>Prediction: ${data.prediction}</h2>
                        <p>Study hours: ${data.study_hours}</p>
                        <p>Attendance: ${data.attendance_percentage}%</p>
                    `;
                }
            } catch (error) {
                result.className = "error";
                result.textContent =
                    "Cannot connect to the server. Check that Flask is running.";
            } finally {
                button.disabled = false;
                button.textContent = "Predict Student Result";
            }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send valid JSON data."}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Provide study_hours and attendance_percentage."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({"error": "Both inputs must be numbers."}), 400

    if study_hours < 0:
        return jsonify({"error": "Study hours cannot be negative."}), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    prediction = model.predict([[study_hours]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
