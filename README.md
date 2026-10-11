# 🎓 Student Pass Prediction System

A beginner-friendly machine learning project that predicts whether a student will pass or fail based on study hours. This project demonstrates model persistence, REST API development using Flask, and a web interface using Streamlit.

## 📌 Project Overview

This project consists of three exercises:

1. **Exercise 1 — Model Persistence:** Train a machine learning model, save it using `joblib`, reload it, and make predictions without retraining.
2. **Exercise 2 — Flask API:** Create a REST API that accepts study hours and attendance percentage and returns a prediction in JSON format.
3. **Exercise 3 — Streamlit Interface:** Build an interactive interface where users enter a student's name and study hours to view the predicted result and a chart.

> **Note:** This project uses sample training data for educational purposes. The model predicts using study hours only. Attendance percentage is validated by the API but does not affect the prediction.

## 🛠️ Technologies Used

- Python
- scikit-learn
- Joblib
- Flask
- Streamlit
- Pandas
- Matplotlib
- PowerShell
- Visual Studio Code (optional)

## 📁 Project Structure

After completing the setup and training, your project should look like this:

```text
student-pass-project/
│
├── .venv/
│   └── Python virtual environment
│
├── train_model.py
│   └── Trains, saves, reloads, and tests the model
│
├── app.py
│   └── Flask REST API
│
├── streamlit_app.py
│   └── Streamlit web interface
│
├── student_pass_model.joblib
│   └── Saved machine learning model
│
├── requirements.txt
│   └── Python package dependencies
│
├── .postman/
│   └── Postman workspace configuration (optional, local use only)
│
├── README.md
│   └── Project documentation
│
└── .gitignore
    └── Files excluded from version control
```

The `.venv` directory and `student_pass_model.joblib` file are generated locally. The model file is created after running the training script.

## ⚙️ Requirements

Before starting, install:

- Python 3.10 or newer (this project is tested with Python 3.12)
- Visual Studio Code (optional)
- Windows PowerShell

Check your Python installation:

```powershell
python --version
```

Check that pip is available:

```powershell
python -m pip --version
```

## 🚀 Installation and Setup

### Step 1 — Open the project folder

Open PowerShell and navigate to your project directory:

```powershell
cd "C:\Users\leych\Wala_Lang\CLO3_ELECT\student-pass-project"
```

### Step 2 — Create a virtual environment

Run:

```powershell
python -m venv .venv
```

If you encounter a permission error because an existing virtual environment is locked or damaged, close running Python processes and VS Code terminals using that environment. Rename or remove the existing `.venv` folder only after confirming it is safe to do so, then recreate it.

### Step 3 — Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the beginning of your PowerShell prompt.

### Step 4 — Install dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the required packages from `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` contains:

```text
scikit-learn>=1.3
joblib>=1.3
Flask>=2.3
streamlit>=1.28
pandas>=2.0
matplotlib>=3.7
```

Alternatively, you can install the packages directly:

```powershell
python -m pip install scikit-learn joblib Flask streamlit pandas matplotlib
```

Verify the installation:

```powershell
python -c "import sklearn, joblib, flask, streamlit, pandas, matplotlib; print('All dependencies installed successfully!')"
```

## 🧠 Exercise 1 — Model Persistence

**File:** `train_model.py`

This exercise demonstrates how to train a logistic regression model, save the trained model, reload it, and make predictions.

### Run the training script

```powershell
python train_model.py
```

The script performs the following tasks:

1. Creates sample study-hours training data.
2. Trains a logistic regression model.
3. Saves the model using `joblib.dump()`.
4. Loads the saved model using `joblib.load()`.
5. Makes three predictions using different study-hour values.

### Expected output

```text
Model saved as student_pass_model.joblib
Saved model loaded successfully!
Study hours: 2 | Prediction: Fail
Study hours: 5 | Prediction: Pass
Study hours: 8 | Prediction: Pass

The saved model can make predictions without being trained again.
```

The exact output for borderline values may vary depending on the training data and model behavior.

### Why is retraining unnecessary?

Training allows the model to learn patterns from the training data. `joblib.dump()` saves the trained model to a file, and `joblib.load()` restores it for later use.

As long as the saved model exists and is compatible with the environment, it can make predictions without repeating the training process.

## 🌐 Exercise 2 — Flask REST API

**File:** `app.py`

The Flask API allows applications to submit student information and receive a prediction through an HTTP request.

### Start the API

First, make sure you have trained the model and generated `student_pass_model.joblib`.

Run:

```powershell
python app.py
```

The API should be available at:

```text
http://127.0.0.1:5000
```

Keep this terminal open while testing the API.

### Available endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Checks whether the API is running |
| POST | `/predict` | Predicts the student's result |

### Test the home endpoint

Open this URL in your browser:

```text
http://127.0.0.1:5000
```

You should receive a JSON response describing the API and its prediction endpoint.

### Test the prediction endpoint

Send a POST request to:

```text
http://127.0.0.1:5000/predict
```

Use this JSON request body:

```json
{
  "study_hours": 5,
  "attendance_percentage": 85
}
```

### Example response

```json
{
  "study_hours": 5.0,
  "attendance_percentage": 85.0,
  "prediction": 1,
  "status": "Pass",
  "note": "Attendance is validated but does not affect this model's prediction."
}
```

The prediction value means:

- `0` — Fail
- `1` — Pass

### Test with PowerShell

Open a second PowerShell terminal in the project folder and activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Submit a request:

```powershell
$body = @{
    study_hours = 5
    attendance_percentage = 85
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

### Test input validation

Try an invalid attendance percentage:

```powershell
$body = @{
    study_hours = 5
    attendance_percentage = 120
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The API should reject this request with an HTTP `400 Bad Request` response because attendance must be between 0 and 100.

Other invalid inputs, such as missing fields and nonnumeric values, should also be rejected.

## 🖥️ Exercise 3 — Streamlit Interface

**File:** `streamlit_app.py`

The Streamlit application provides a simple web interface for predicting student results.

### Start the interface

Open a third PowerShell terminal, navigate to the project folder, and activate the virtual environment:

```powershell
cd "C:\Users\leych\Wala_Lang\CLO3_ELECT\student-pass-project"

.\.venv\Scripts\Activate.ps1
```

Run the application:

```powershell
python -m streamlit run streamlit_app.py
```

Open the local address shown in the terminal. It is usually:

```text
http://localhost:8501
```

### Features

The interface includes:

- Student name input
- Study-hours input
- Predict Result button
- Pass or fail prediction
- Prediction explanation
- Chart showing predicted results for different study-hour values

### How to use it

1. Open the Streamlit application in your browser.
2. Enter the student's name.
3. Enter the number of study hours.
4. Click **Predict Result**.
5. Review the prediction and chart.

The application uses the saved model, so you do not need to train the model every time you open the interface.

## 🔧 Troubleshooting

### 1. Model file not found

**Error:**

```text
FileNotFoundError: student_pass_model.joblib
```

**Solution:** Run the training script first:

```powershell
python train_model.py
```

Ensure that the saved model and the Python scripts are in the same project directory.

### 2. Virtual environment permission error

**Solution:**

- Close terminals and processes using the environment.
- Check whether `.venv` already exists.
- If it is damaged, rename or remove it after confirming it is safe to do so.
- Recreate it using `python -m venv .venv`.

### 3. Flask port already in use

If port 5000 is occupied, stop the other application using that port or change the port in `app.py`.

### 4. PowerShell cannot activate the virtual environment

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then run:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Missing Python packages

Activate the virtual environment and reinstall the dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 6. Streamlit cannot find the model

Make sure you have run `train_model.py` before starting Streamlit and that both files are in the same folder.



### HOW TO RUN - Student Pass Prediction Project
=============================================

Open PowerShell in the project folder:
  cd "C:\Users\leych\Wala_Lang\CLO3_ELECT\student-pass-project"

Activate the virtual environment (run this in EVERY terminal you open):
  .\.venv\Scripts\Activate.ps1

If PowerShell blocks activation, run first:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


STEP 0 - Install dependencies (once only)
-----------------------------------------
Terminal: any (run once)
  python -m pip install -r requirements.txt


STEP 1 - Train the model (once, before the apps)
-------------------------------------------------
Terminal: Terminal 1
  python train_model.py

Creates student_pass_model.joblib. Run again only if you delete the model file.


STEP 2 - Flask REST API
-----------------------
Terminal: Terminal 2 (SEPARATE from Terminal 1)
  python app.py

API runs at http://127.0.0.1:5000
Keep this terminal open while testing.

Test it in another terminal (Terminal 3, SEPARATE):
  .\.venv\Scripts\Activate.ps1
  $body = @{ study_hours = 5; attendance_percentage = 85 } | ConvertTo-Json
  Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body $body

Or open in a browser: http://127.0.0.1:5000


STEP 3 - Streamlit web interface
---------------------------------
Terminal: Terminal 3 (SEPARATE from the Flask terminal)
  python -m streamlit run streamlit_app.py

Open http://localhost:8501 in your browser.


SUMMARY - how many terminals
-----------------------------
Terminal 1: python train_model.py        (run once, then it can be closed)
Terminal 2: python app.py                (Flask API - keep running)
Terminal 3: streamlit run streamlit_app.py (Streamlit UI - keep running)

- Step 1 must run BEFORE steps 2 and 3 (the model file must exist).
- Steps 2 and 3 MUST be in separate terminals, both running at the same time.
- Steps 2 and 3 must NOT be in the same terminal as step 1 if you still need step 1 there.
- Each new terminal needs the activate command: .\.venv\Scripts\Activate.ps1


STOPPING
--------
Press Ctrl + C in the terminal to stop Flask or Streamlit.
