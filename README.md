# IT_Elect3_Exercise

This folder contains solutions for the three exercises in `Exercise.html`.

## Files

- `train_model.py` — trains a simple model, saves it as `student_pass_model.joblib`, reloads it, and predicts results for three study-hour values.
- `app.py` — Flask API that accepts `study_hours` and `attendance_percentage`, validates the attendance percentage, and returns JSON.
- `streamlit_app.py` — Streamlit interface with a student name, study-hours input, prediction button, result explanation, and bonus chart.
- `requirements.txt` — Python packages needed by the project.

## Setup

Use Python 3.10 or newer if available. Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows Command Prompt**
```bat
.venv\Scripts\activate
```

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

Install packages:

```bash
python -m pip install -r requirements.txt
```

## Exercise 1: Train and save the model

```bash
python train_model.py
```

This creates `student_pass_model.joblib` in this folder and prints predictions for 2, 5, and 8 study hours.

**Why doesn't it need to train again?** The saved file contains the model's learned parameters. Loading it with `joblib.load()` restores the trained model, which can then make predictions for new inputs. Retraining is needed when you want to fit the model again using new or changed training data.

## Exercise 2: Run and test the Flask API

First run `python train_model.py` if you have not already done so. Start the API:

```bash
python app.py
```

Send a POST request to `http://127.0.0.1:5000/predict` with this JSON body:

```json
{
  "study_hours": 5,
  "attendance_percentage": 90
}
```

Using `curl.exe` on Windows:

```bat
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 5, \"attendance_percentage\": 90}"
```

The response includes both inputs and the predicted class. Attendance is validated and returned, but it does **not** affect the current prediction because this example model is trained with study hours only. To make attendance influence predictions, update the training data and train a model with both features.

## Exercise 3: Run the Streamlit interface

```bash
streamlit run streamlit_app.py
```

The browser page lets you enter a student's name and study hours, then displays a predicted result, an explanation, and a simple chart.

## Important note

The training examples in `train_model.py` are illustrative sample data, not a validated real-world student dataset. Replace them with the dataset supplied by your instructor if one was provided. Predictions are educational examples only and are not guarantees of actual academic outcomes.
