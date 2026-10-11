# Student Success Dashboard

A classroom demonstration based on the attached **Final-Lesson** and **Exercise** materials: model persistence with Joblib, a Streamlit interface, a Flask prediction API, and a chart comparing study hours with model predictions.

## Features
- Clean responsive Streamlit dashboard with student name and study-hours inputs.
- Predict button, Pass/Fail result, estimated pass probability, and explanation.
- Interactive Plotly curve showing study hours versus estimated pass probability.
- Saved model loaded from `models/student_pass_model.joblib`; predictions do not retrain it.
- Flask `POST /predict` endpoint returning JSON and validating study hours and optional attendance (0–100).

> **Important:** The training data in `train_model.py` is illustrative sample data for learning. The model uses **study hours only**. Attendance is validated and echoed by the API but does not affect predictions unless you add attendance to the training dataset and retrain the model. Do not use this demo to make real academic decisions.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
```

## Run the Streamlit UI

```powershell
streamlit run streamlit_app.py
```

## Run the Flask API

Open another PowerShell terminal in the same project folder:

```powershell
python app.py
```

Check health at `http://127.0.0.1:5000/health`.

Example request using `curl.exe`:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"
```

Example response (the prediction depends on the trained model):

```json
{
  "study_hours": 5.0,
  "attendance_percentage": 90.0,
  "prediction": "Pass",
  "model_features": ["study_hours"],
  "note": "Attendance is validated and returned but does not affect this model's prediction."
}
```

## Git safety note
The provided `Git-TXT.txt` includes a GitHub URL with what appears to be an access token embedded in it. Do **not** reuse or commit that URL. If it was a real token, revoke it in GitHub settings and create a new token if needed. Use a clean remote URL instead:

```powershell
git remote add origin https://github.com/OWNER/REPOSITORY.git
```
