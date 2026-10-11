# Student ML Model Persistence and Deployment

A beginner-friendly project based on the `Final-Lesson.html` and `Exercise.html` lesson materials.

## Project files

- `train_model.py` — trains, saves, and loads the model; demonstrates three predictions.
- `app.py` — Flask API endpoint that accepts study hours and attendance percentage.
- `streamlit_app.py` — interactive student prediction interface with a chart.
- `requirements.txt` — Python packages needed to run the project.
- `student_pass_model.joblib` — created the first time the model is trained.

## Before you start

Install:
1. Python 3.10 or newer (check with `py --version` in Windows PowerShell).
2. Visual Studio Code.
3. The Python extension in VS Code (Microsoft).

## Step-by-step setup on Windows

### 1. Extract the ZIP
Extract `student_ml_deployment_project.zip` to a folder you can find, such as your Desktop.

### 2. Open the folder in VS Code
Open VS Code → **File** → **Open Folder...** → choose `student_ml_deployment_project`.

### 3. Open a terminal
In VS Code, choose **Terminal → New Terminal**.

### 4. Create a virtual environment
Run this in the VS Code PowerShell terminal:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt as the terminal and run:

```bat
.venv\Scripts\activate.bat
```

### 5. Install the packages

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Wait until installation finishes.

### 6. Train and save the model

```powershell
py train_model.py
```

This creates `student_pass_model.joblib` and prints predictions for 2, 5, and 8 study hours. The exact model output is produced by the trained sample model.

### 7. Run the Streamlit interface

In the terminal:

```powershell
py -m streamlit run streamlit_app.py
```

A browser tab should open. Enter a name, choose study hours, and click **Predict**. To stop the app, return to the terminal and press `Ctrl+C`.

### 8. Run the Flask API

Stop Streamlit first with `Ctrl+C`, then run:

```powershell
py app.py
```

Keep this terminal running. The API is at `http://127.0.0.1:5000`.

Open `http://127.0.0.1:5000` in your browser to check the API's home response.

### 9. Test the API with curl.exe

Open a **second VS Code terminal** while Flask is still running, then paste:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"
```

It should return JSON containing both input values and a prediction. The prediction is generated from study hours only.

Try an invalid attendance value to test validation:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":150}"
```

The API should return an error because attendance must be between 0 and 100.

## How this completes the exercises

### Exercise 1 — Model Persistence
- Saves the trained model with `joblib.dump()`.
- Loads it with `joblib.load()`.
- Demonstrates three predictions.
- The model does not need to be trained again for predictions because the learned model is saved in a file and loaded when needed.

### Exercise 2 — Flask API
- Accepts `study_hours` and `attendance_percentage`.
- Checks attendance is from 0 to 100.
- Returns both values and the prediction as JSON.
- Can be tested using `curl.exe` or Postman.
- Important: attendance is not used by the current model. To make it affect the result, update the training data and train a model with both features.

### Exercise 3 — Streamlit Interface
- Includes a student name input.
- Includes a study-hours input.
- Has a Predict button and result message.
- Explains the prediction.
- Includes a simple chart as the bonus task.

## Troubleshooting

- **`py` is not recognized:** Install Python from python.org and select the option to add Python to PATH, then reopen VS Code.
- **`No module named ...`:** Activate `.venv` and run `py -m pip install -r requirements.txt` again.
- **Browser page does not open:** Copy the local URL printed in the terminal into your browser.
- **Port already in use:** Stop the other running app with `Ctrl+C` before starting it again.

## Note
The dataset is tiny and artificial for learning purposes only. The prediction is not an actual guarantee of whether a student will pass or fail.
