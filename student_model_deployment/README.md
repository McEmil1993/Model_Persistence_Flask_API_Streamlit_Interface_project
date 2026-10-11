# Student Pass Prediction — Model Persistence and Deployment

This project contains solutions for the three exercises in the lesson.

## Requirements

- Python 3.10 or newer
- pip

## 1. Set up the project

Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate the environment.

**Windows Command Prompt:**
```bat
.venv\Scripts\activate
```

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

## 2. Train and save the model

```bash
python train_model.py
```

This creates `student_pass_model.joblib`. The model is trained with study hours only.

## Exercise 1 — Model Persistence

```bash
python exercise1_persistence.py
```

The script loads the saved model and predicts results for 2, 4, and 7 study hours without retraining.

**Explanation:** The trained model is saved in a file with `joblib.dump()` and reused with `joblib.load()`. Since the trained model is loaded from disk, it does not need to run training again for each prediction.

## Exercise 2 — Flask API

Start the API:

```bash
python app.py
```

It runs at `http://127.0.0.1:5000`.

Send a POST request to `http://127.0.0.1:5000/predict` with JSON:

```json
{
  "study_hours": 5,
  "attendance_percentage": 90
}
```

Example using Windows Command Prompt or PowerShell with `curl.exe`:

```bat
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"
```

The response contains both input values and the prediction. Attendance is validated from 0 through 100, but it does not affect the prediction because the model is trained using study hours only.

To test invalid attendance, send `attendance_percentage` as `101`; the API should return HTTP 400 with an error message.

## Exercise 3 — Streamlit Interface

In a second terminal, activate the same virtual environment and run:

```bash
streamlit run streamlit_app.py
```

The app opens in a browser. Enter a student name, choose study hours, and click **Predict Result**. The app displays the predicted class, an explanation, and a bonus chart.

## Important note about the sample data

The training records in `train_model.py` are illustrative sample data because no separate dataset was included with the exercise. If your instructor provided a dataset or starter `train_model.py`, use those official training records to match the class lesson exactly. This sample model uses only study hours, as stated in Exercise 2.

## GitHub

Before pushing, review `.gitignore` and never upload passwords, access tokens, or `.env` files. The `.joblib` model is ignored because it can be recreated by running `python train_model.py`.
