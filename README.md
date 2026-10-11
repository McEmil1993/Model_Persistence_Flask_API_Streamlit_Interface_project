# Student Pass Predictor

A beginner-friendly machine learning project that predicts whether a student will **pass or fail** based on study hours and attendance.

**What you will learn:**

1. **Model persistence:** train a model once, save it to a file, and reuse it without retraining.
2. **Flask API:** send data to a web endpoint and get a prediction back as JSON.
3. **Streamlit app:** a simple web page where users type in a name and study hours.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Before You Start](#before-you-start)
3. [Installation](#installation)
4. [Running the Project](#running-the-project)
5. [Testing the API with Postman](#testing-the-api-with-postman)
6. [Testing the API with curl](#testing-the-api-with-curl)
7. [Understanding the Terminal Output (Debugging)](#understanding-the-terminal-output-debugging)
8. [Troubleshooting](#troubleshooting)
9. [How It Works](#how-it-works)
10. [Glossary](#glossary)

---

## Project Structure

```
student_prediction/
├── train_model.py                 # Trains the study-hours model and saves it (Exercise 1)
├── train_model_attendance.py      # Trains the study-hours + attendance model (Exercise 2)
├── app.py                         # Flask API (Exercise 2)
├── streamlit_app.py               # Streamlit web interface (Exercise 3)
├── requirements.txt               # List of Python packages to install
├── .gitignore                     # Files Git should ignore
└── README.md                      # This guide
```

Running the training scripts creates two model files in the same folder:

| File                                   | Created by                  | Used by              |
| -------------------------------------- | --------------------------- | -------------------- |
| `student_pass_model.joblib`            | `train_model.py`            | `streamlit_app.py`   |
| `student_pass_model_attendance.joblib` | `train_model_attendance.py` | `app.py` (Flask API) |

---

## Before You Start

You need these installed:

| Tool                                           | Why                             | How to check       |
| ---------------------------------------------- | ------------------------------- | ------------------ |
| **Python 3.9 or newer**                        | Runs the project                | `python --version` |
| **Git**                                        | Downloads the project           | `git --version`    |
| **VS Code** (recommended)                      | Editor with a built-in terminal |                    |
| **Postman** (or the Postman VS Code extension) | Tests the API                   |                    |

> **Windows tip:** When installing Python, tick **"Add Python to PATH"** on the first screen. If `python` is not recognized later, try `py` instead.

---

## Installation

Open a terminal (in VS Code: **Terminal → New Terminal**) and follow each step in order.

### Step 1: Download the project

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### Step 2: Create a virtual environment

A virtual environment is a private folder of packages for this project only, so it will not clash with other projects.

**Windows**

```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

You will know it worked when `(venv)` appears at the start of your terminal line:

```
(venv) PS C:\Projects\student_prediction>
```

> **PowerShell says "running scripts is disabled"?** Run this once, then activate again:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

> **Using VS Code?** Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick the one inside `venv`.

### Step 3: Install the packages

```bash
pip install -r requirements.txt
```

This installs numpy, pandas, scikit-learn, joblib, Flask, and Streamlit. It can take a few minutes.

---

## Running the Project

Run these in order. The apps need the model files to exist first.

### Step 1: Train and save the models

```bash
python train_model.py
python train_model_attendance.py
```

Expected output from the first script:

```
Model saved to student_pass_model.joblib
Study hours:    2 -> FAIL (pass probability: 0.02)
Study hours:  5.5 -> PASS (pass probability: 0.50)
Study hours:    9 -> PASS (pass probability: 0.98)
```

The three predictions come from the model **loaded from the file**. It is not trained again.

### Step 2: Start the Flask API

```bash
python app.py
```

You should see:

```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

**Leave this terminal open.** The API only works while this is running. To stop it, press `Ctrl+C`.

Open `http://127.0.0.1:5000` in your browser to confirm it is alive. You should see:

```json
{
  "message": "Student Pass Predictor API",
  "usage": "POST /predict with JSON: {\"study_hours\": 7, \"attendance_percentage\": 85}"
}
```

### Step 3: Start the Streamlit app

Open a **second terminal** (click the **+** in the VS Code terminal panel), activate the virtual environment again, then run:

```bash
streamlit run streamlit_app.py
```

Your browser opens at `http://localhost:8501`. If `streamlit` is not recognized, use:

```bash
python -m streamlit run streamlit_app.py
```

The Flask API and the Streamlit app are separate. You can run either one without the other.

---

## Testing the API with Postman

### The API at a glance

|               |                                                   |
| ------------- | ------------------------------------------------- |
| **URL**       | `http://127.0.0.1:5000/predict`                   |
| **Method**    | `POST` (a `GET` with query parameters also works) |
| **Body type** | JSON                                              |

| Field                   | Type   | Rule              |
| ----------------------- | ------ | ----------------- |
| `study_hours`           | number | 0 or more         |
| `attendance_percentage` | number | between 0 and 100 |

### Send your first request

1. Make sure `python app.py` is running.
2. In Postman, click **New HTTP Request**.
3. Set the method to **POST** (use the dropdown at the left of the URL bar).
4. Enter the URL: `http://127.0.0.1:5000/predict`
5. Click the **Body** tab.
6. Select **raw**, then choose **JSON** from the dropdown on the right.
7. Paste this into the body box:
   ```json
   { "study_hours": 7, "attendance_percentage": 85 }
   ```
8. Click **Send**.

You should see this in the response area:

```json
{
  "attendance_percentage": 85.0,
  "pass_probability": 0.989,
  "prediction": "pass",
  "study_hours": 7.0
}
```

Status: **200 OK**.

<!-- Add your screenshot: ![Successful Postman request](docs/postman-success.png) -->

### Test more cases

Edit the body, click **Send**, and compare the status code:

| Test                   | Body                                                  | Expected status |
| ---------------------- | ----------------------------------------------------- | --------------- |
| Normal (pass)          | `{"study_hours": 7, "attendance_percentage": 85}`     | 200             |
| Low values (fail)      | `{"study_hours": 2, "attendance_percentage": 50}`     | 200             |
| Lowest attendance      | `{"study_hours": 5, "attendance_percentage": 0}`      | 200             |
| Highest attendance     | `{"study_hours": 5, "attendance_percentage": 100}`    | 200             |
| Attendance too high    | `{"study_hours": 3, "attendance_percentage": 101}`    | 400             |
| Attendance negative    | `{"study_hours": 3, "attendance_percentage": -5}`     | 400             |
| Text instead of number | `{"study_hours": "abc", "attendance_percentage": 80}` | 400             |
| Missing field          | `{"study_hours": 5}`                                  | 400             |
| Negative hours         | `{"study_hours": -1, "attendance_percentage": 80}`    | 400             |

Example error response (attendance 101):

```json
{ "error": "attendance_percentage must be between 0 and 100" }
```

### Test with a GET request (optional)

1. Change the method to **GET**.
2. Open the **Params** tab and add two rows: `study_hours` = `7` and `attendance_percentage` = `85`.
3. Click **Send**.

Postman builds the URL: `http://127.0.0.1:5000/predict?study_hours=7&attendance_percentage=85`

> A GET request with **no** parameters returns a 400 error. That is normal: there are no numbers to predict from.

### Save your requests (optional)

Click **Save**, create a collection named "Student Pass API", and give each request a clear name such as "Attendance above 100". If you see a dot next to the tab name, the request has unsaved changes.

### Understanding status codes

| Code                     | Meaning              | What to do                                       |
| ------------------------ | -------------------- | ------------------------------------------------ |
| **200**                  | Success              | Read the prediction in the body                  |
| **400**                  | Bad input            | Read the `error` message and fix your request    |
| **404**                  | URL not found        | Check the URL ends with `/predict`               |
| **405**                  | Wrong method         | Use POST or GET                                  |
| **500**                  | Server crashed       | Read the error in the terminal running `app.py`  |
| "Could not get response" | Server not reachable | Start `python app.py` and check the URL and port |

---

## Testing the API with curl

Postman is not required. You can test from a terminal too.

**Windows (use `curl.exe`, not `curl`, so PowerShell does not use its own alias):**

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 7, \"attendance_percentage\": 85}"
```

**macOS / Linux:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 7, "attendance_percentage": 85}'
```

---

## Understanding the Terminal Output (Debugging)

The terminal running `python app.py` is your best debugging tool. Debug mode is on, so every request prints information.

### A successful request

```
[DEBUG] POST Content-Type=application/json data={'study_hours': 7, 'attendance_percentage': 85}
[DEBUG] prediction=1 probability=0.989
127.0.0.1 - - [11/Oct/2026 07:14:42] "POST /predict HTTP/1.1" 200 -
```

| Line                             | Meaning                                             |
| -------------------------------- | --------------------------------------------------- |
| `Content-Type=application/json`  | Postman sent the body as JSON (correct)             |
| `data={...}`                     | The exact values the server received                |
| `prediction=1 probability=0.989` | The model predicted pass (1) with 98.9% probability |
| `"POST /predict HTTP/1.1" 200`   | Request succeeded                                   |

### Reading a 400 error

Check the `[DEBUG]` line above it:

| What you see              | Meaning                            | Fix                                                                   |
| ------------------------- | ---------------------------------- | --------------------------------------------------------------------- |
| `Content-Type=text/plain` | Body type is wrong                 | Set Body to **raw → JSON** in Postman                                 |
| `data={}`                 | The server received no usable data | Check for single quotes, trailing commas, or a GET with no parameters |
| `data={'study_hours': 5}` | A field is missing                 | Add `attendance_percentage`                                           |

### Normal messages you can ignore

| Message                                 | Why it is harmless                                |
| --------------------------------------- | ------------------------------------------------- |
| `WARNING: This is a development server` | It only means "don't use this on a public server" |
| `GET /favicon.ico 404`                  | Browsers ask for a tab icon automatically         |
| `Debugger PIN: ...`                     | Only used with browser error pages                |

> **Never run with `debug=True` on a public server.** It is for your own computer only.

---

## Troubleshooting

| Problem                                    | Fix                                                                                                        |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `python` is not recognized                 | Reinstall Python with "Add to PATH" ticked, or use `py`                                                    |
| `ModuleNotFoundError: No module named ...` | Activate the virtual environment, then run `pip install -r requirements.txt`                               |
| `FileNotFoundError: ...joblib`             | Run both training scripts first, from inside the project folder                                            |
| `'streamlit' is not recognized`            | Activate the virtual environment, or use `python -m streamlit run streamlit_app.py`                        |
| Postman: "Could not get response"          | `python app.py` is not running, or the URL or port is wrong                                                |
| 404 on `POST /`                            | The URL is missing `/predict`                                                                              |
| 400 "must be numbers"                      | Body must be raw → JSON, with plain double quotes and numeric values                                       |
| 400 "Missing field(s)"                     | The body needs both `study_hours` and `attendance_percentage`                                              |
| Postman shows an old response              | Click **Send** again                                                                                       |
| Port 5000 already in use                   | Change the last line of `app.py` to `app.run(debug=True, use_reloader=False, port=5001)` and use port 5001 |
| Code changes are not showing               | Stop the server with `Ctrl+C` and run `python app.py` again                                                |
| Warnings after upgrading scikit-learn      | Run both training scripts again to rebuild the model files                                                 |

---

## How It Works

1. **Training:** `fit()` finds the pattern in the example data and learns numbers called coefficients. This only happens in the training scripts.
2. **Saving:** `joblib.dump()` writes the trained model to a `.joblib` file.
3. **Loading:** `joblib.load()` restores the same trained model, so no retraining is needed.
4. **Predicting:** the model is a **logistic regression**. It turns the inputs into a pass probability. Above 50% it predicts **pass**, otherwise **fail**.

> The training data is a small example set made for learning. To change the results, edit the data in the training scripts and run them again.

---

## Glossary

| Term                    | Meaning                                                                         |
| ----------------------- | ------------------------------------------------------------------------------- |
| **API**                 | A way for programs to talk to each other, here by sending data to a web address |
| **Endpoint**            | One specific address of an API, such as `/predict`                              |
| **JSON**                | A text format for data, like `{"study_hours": 7}`                               |
| **Model**               | A program that learned a pattern from data                                      |
| **Training**            | Teaching the model using example data                                           |
| **Prediction**          | The model's answer for new input                                                |
| **Virtual environment** | A private set of packages for one project                                       |
| **Status code**         | A number saying whether a request worked (200) or failed (400, 404, 500)        |

---

## Tech Stack

Python, scikit-learn, joblib, Flask, Streamlit, pandas, NumPy
