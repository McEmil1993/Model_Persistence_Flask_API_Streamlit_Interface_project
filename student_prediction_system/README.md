# Student Prediction System

This project is organized as requested:
- **Exercise 1 — Model:** `train_model.py` trains and saves `student_pass_model.joblib`.
- **Exercise 2 — Backend:** `app.py` provides a Flask JSON API.
- **Exercise 3 — Frontend:** `streamlit_app.py` provides the Streamlit interface.

## 1. Setup

Use Python 3.10 or newer. Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Install packages:
```bash
python -m pip install -r requirements.txt
```

## 2. Exercise 1: train and save the model

```bash
python train_model.py
```

This creates `student_pass_model.joblib`, loads it again, and prints predictions for 2, 5, and 8 study hours. The included data is illustrative; use your class dataset if one was provided.

**Why no retraining is needed for each prediction:** `joblib.dump()` saves the fitted model, including its learned decision rules. `joblib.load()` restores that fitted model so it can predict new inputs. Retrain only when you want to change/update the model or its training data.

## 3. Exercise 2: run the Flask backend

Keep the generated `.joblib` file in the same folder. Start Flask:

```bash
python app.py
```

Test in a second terminal with `curl.exe`:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"
```

Example response:
```json
{
  "study_hours": 5.0,
  "attendance_percentage": 90.0,
  "prediction": "Pass",
  "note": "Attendance is validated and returned, but does not affect the prediction because this model uses study_hours only."
}
```

Test invalid attendance:
```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":120}"
```
It should return HTTP 400 and an error message.

## 4. Exercise 3: run the Streamlit frontend

Leave Flask running. Open another terminal in the project folder, activate the same virtual environment, then run:

```bash
streamlit run streamlit_app.py
```

Streamlit opens the interface in your browser. Enter a student name, study hours, and attendance, then click **Predict**.

## Important limitation

The current model is trained with **study hours only**. The API accepts and validates attendance from 0 to 100 and returns it in JSON, but attendance does not affect the prediction yet. To make attendance influence the prediction, update the training dataset and train a model using both `study_hours` and `attendance_percentage`.
