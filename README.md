# Student Pass Predictor

A small machine-learning web system that predicts whether a student will pass,
based on study hours and attendance.

## Setup (Windows)
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Run
    python train_model.py              # Exercise 1: save/load model, 3 predictions
    python train_attendance_model.py   # Exercise 2: train with attendance
    python app.py                      # Web app + API  -> http://127.0.0.1:5000
    streamlit run streamlit_app.py     # Exercise 3 interface
    python -m pytest -v                # Run automated tests

## API
POST /predict   {"study_hours": 7, "attendance_percentage": 85}

    curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 7, \"attendance_percentage\": 85}"
