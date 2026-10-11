# StudyMate AI

A student-created machine-learning demonstration for IT Elective 3. It includes model persistence, a Flask JSON API, and a Streamlit interface.

## Requirements

- Python 3.10 or newer
- pip

## Setup on Windows PowerShell

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Exercise 1 — Model persistence

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The script trains a small Decision Tree model, saves it as `studymate_model.joblib`, loads the saved file, and makes three example predictions.

## Exercise 2 — Flask API

Start the server:

```powershell
.\.venv\Scripts\python.exe app.py
```

In a second PowerShell terminal, test a valid request:

```powershell
$body = '{"study_hours":5,"attendance_percentage":90}'
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType 'application/json' -Body $body
```

Test invalid attendance:

```powershell
$body = '{"study_hours":5,"attendance_percentage":120}'
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType 'application/json' -Body $body
```

The second request should be rejected with HTTP 400.

## Exercise 3 — Streamlit interface

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

## Model note

The training examples are small, manually prepared synthetic data for demonstrating the programming workflow. They are not real student records and should not be used to judge actual academic performance.
