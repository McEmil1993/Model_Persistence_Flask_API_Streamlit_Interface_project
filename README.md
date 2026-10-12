# Student Prediction Exercises — Version 2

Includes model persistence, a Flask API, and a Streamlit interface.

Python only — no `.bat` files. All setup and runs use Python.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup.py
```

Or without a venv:

```powershell
python setup.py
```

If `python` is not recognized, replace `python` with `py`.

## Train only

```powershell
python run_train.py
```

## Run Flask API

```powershell
python run_flask.py
```

In another terminal, test the endpoint:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"study_hours":5,"attendance_percentage":90}'
```

## Run Streamlit

Stop Flask with Ctrl+C, then run:

```powershell
python run_streamlit.py
```

Open the local URL printed in the terminal.

## Attendance note

The sample model is trained using study hours only. The API accepts and validates attendance from 0 to 100, but attendance does not affect the prediction unless the model is retrained using attendance as an input feature.

## Files

- `train_model.py`: trains, saves, reloads, and tests the model.
- `app.py`: Flask API with input validation.
- `streamlit_app.py`: student input, prediction, explanation, and chart.
- `setup.py`: Python-only setup (pip upgrade + install requirements + train model).
- `run_train.py`: Python-only runner for training.
- `run_flask.py`: Python-only runner for Flask.
- `run_streamlit.py`: Python-only runner for Streamlit.
- `requirements.txt`: required Python packages.
