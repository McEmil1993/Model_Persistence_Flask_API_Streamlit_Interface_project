# Student Pass Prediction — Python Activity

A classroom project with three exercises and a redesigned Streamlit dashboard.

1. **Model persistence** — train a sample pass/fail model and save it with `joblib`.
2. **Flask API** — accept study hours and attendance percentage and return a JSON prediction.
3. **Streamlit dashboard** — a responsive, classroom-inspired interface with warm paper tones, evergreen panels, amber accents, student inputs, prediction results, a probability chart, and a comparison table.

## Setup (Windows)

Open PowerShell in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
```

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Run Exercise 1 — Train and save the model

```powershell
python train_model.py
```

This creates `student_pass_model.joblib` in the project folder.

## Run Exercise 2 — Flask API

```powershell
python app.py
```

The API runs at `http://127.0.0.1:5000`. Test from another terminal:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{"study_hours":5,"attendance_percentage":85}"
```

## Run Exercise 3 — Streamlit dashboard

```powershell
python -m streamlit run streamlit_app.py
```

Open the local URL printed in the terminal. Enter a student name, study hours, and attendance, then click **Generate student estimate**.

## Important model limitation

The model uses **study hours only**. Attendance is accepted and validated from 0 to 100, but it is recorded for the activity and does not influence the prediction. The dataset is small sample/demo data, so results are illustrative and must not be treated as an official academic assessment.

## Project files

- `train_model.py` — trains and saves the model
- `app.py` — Flask JSON API
- `streamlit_app.py` — redesigned Streamlit dashboard
- `requirements.txt` — required packages
- `README.md` — setup and run instructions
- `run_dashboard.bat` — Windows shortcut to launch the dashboard
