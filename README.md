# 🎓 Student Pass/Fail Prediction — ML Exercise

A complete machine learning mini-project that trains a **Logistic Regression** model to predict whether a student will **pass or fail** based on **study hours and attendance**. It covers model persistence, a REST API, and an interactive web UI.

---

## 📂 Project Structure

```
student-pass-prediction/
├── train_model.py          # Exercise 1 — Train, save, load, predict
├── app.py                  # Exercise 2 — Flask REST API
├── streamlit_app.py        # Exercise 3 — Streamlit web interface
├── requirements.txt        # Python dependencies
└── README.md               # ← You are here
```

---

## ⚙️ Prerequisites

- **Python 3.8+** installed and available on your PATH.
- **pip** (comes with Python).

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Exercise 1 — Train & persist the model

```bash
python train_model.py
```

**What it does:**
1. Trains a Logistic Regression model on student data with **two features**: `study_hours` and `attendance_percentage`.
2. Saves the trained model to `student_pass_model.joblib`.
3. Loads the model back from disk.
4. Makes sample predictions and prints pass probabilities.
5. Explains why the model doesn't need retraining after loading.

> [!IMPORTANT]
> You **must** run this step first — both the Flask API and the Streamlit app depend on the saved model file.

### 3. Exercise 2 — Run the Flask API

```bash
python app.py
```

The server starts at `http://127.0.0.1:5000`. To test it, send a POST request:

**With curl (Windows `curl.exe` / macOS / Linux):**
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 85}"
```

**With PowerShell:**
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"study_hours": 6, "attendance_percentage": 85}'
```

> [!TIP]
> In PowerShell, use **single quotes** around the JSON body (as above). Double-quoted bodies need the inner quotes escaped with backslashes.

**Expected JSON response:**
```json
{
  "study_hours": 6.0,
  "attendance_percentage": 85.0,
  "prediction": 1,
  "result": "Pass",
  "probability_pass": 0.9962
}
```

**Other endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API info / usage (so opening the URL in a browser works) |
| `/health` | GET | Health check, returns `{"status": "ok", ...}` |
| `/predict` | POST | Pass/fail prediction from JSON body |

Both `study_hours` **and** `attendance_percentage` influence the prediction.

### 4. Exercise 3 — Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Opens an interactive web page (usually at `http://localhost:8501`) where you can:
- Enter a student name
- Adjust study hours **and attendance** with sliders
- Click **Predict** to see the result + pass probability
- View an explanation of how the model works
- See a bar chart comparing study hours with predicted classes (at the chosen attendance)

---

## 📖 How the Flow Works

```
┌─────────────────────┐
│  1. train_model.py   │   Trains the model on [study_hours, attendance] data
│                      │   and saves it as a .joblib file.
└──────────┬──────────┘
           │  student_pass_model.joblib
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│   2. app.py          │     │ 3. streamlit_app.py  │
│   (Flask API)        │     │ (Streamlit Web UI)   │
│                      │     │                      │
│ Loads the saved model│     │ Loads the saved model │
│ Exposes /predict     │     │ Sliders + prediction  │
│ endpoint (POST)      │     │ + chart               │
└──────────────────────┘     └──────────────────────┘
```

1. **Training phase** (`train_model.py`):
   - A Logistic Regression model is fitted on labelled data ([study hours, attendance] → pass/fail).
   - The learned parameters (weights & bias) are serialized to disk via `joblib.dump()`.

2. **Serving phase** (`app.py` or `streamlit_app.py`):
   - The serialized model is loaded with `joblib.load()` — no retraining needed.
   - New input data is fed to `model.predict()` / `model.predict_proba()` and the result is returned to the user.

---

## 🧪 Quick Test Checklist

| Step | Command | Expected Outcome |
|------|---------|-------------------|
| Install deps | `pip install -r requirements.txt` | All packages install successfully |
| Train model | `python train_model.py` | Prints training accuracy + predictions; creates `student_pass_model.joblib` |
| Flask API | `python app.py` | Server runs on port 5000 (no reload loop); POST returns JSON with `result` and `probability_pass` |
| Streamlit | `streamlit run streamlit_app.py` | Browser opens with interactive UI (name + 2 sliders) |

---

## 🔧 Troubleshooting

- **`Model file not found`**: run `python train_model.py` first, from the project folder.
- **Flask server keeps restarting / never responds**: fixed by `use_reloader=False` in `app.py`. If you still see a watchdog reload loop, update `watchdog` or set `debug=False`.
- **`Request body must be valid JSON` (PowerShell)**: make sure the `-Body` uses single quotes: `'{"study_hours": 6, "attendance_percentage": 85}'`.
- **Old 1-feature model file**: delete `student_pass_model.joblib` and re-run `python train_model.py` to get the current 2-feature model. (The API and Streamlit app also fall back gracefully if they find an old file.)

---

## 📝 License

This project is for educational purposes.
