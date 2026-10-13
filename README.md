# Model Persistence, Flask API, and Streamlit Interface

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Exercise 1: model persistence

```powershell
python train_model.py
```

This creates `student_pass_model.joblib`, reloads it, and makes predictions for three different study-hour values.

## Exercise 2: Flask API

```powershell
python app.py
```

Test it from PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 7, \"attendance_percentage\": 90}"
```

Attendance is checked and returned in the response. The model currently predicts from study hours only, as required by the exercise note.

## Exercise 3: Streamlit interface

```powershell
streamlit run streamlit_app.py
```
