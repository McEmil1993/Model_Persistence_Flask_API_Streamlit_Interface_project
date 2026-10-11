# Student Pass Prediction Exercises

This sample project demonstrates saving/loading a scikit-learn model, exposing a Flask JSON API, and building a Streamlit interface. The included `students.csv` is small synthetic example data with study hours, attendance percentage, and pass labels so the exercises run as-is. Its labels are illustrative and are not a real student assessment.

## Setup

From this folder, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Exercise 1: Persist the model

```powershell
py train_model.py
```

This trains logistic regression from `students.csv` using study hours and attendance percentage, writes `student_pass_model.joblib` with `joblib.dump()`, then reloads it with `joblib.load()` and predicts for three example students. The saved file contains the fitted model parameters. `predict()` can use those parameters directly, so training is needed only when fitting or updating the model, not for each prediction.

## Exercise 2: Flask API

Start the API:

```powershell
py app.py
```

Send a request from another PowerShell window using `curl.exe`:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours":6,"attendance_percentage":92}'
```

The response includes both supplied inputs and the predicted class. Attendance must be between 0 and 100. This version trains the model using both `study_hours` and `attendance_percentage`, so both inputs are used in its prediction.

Example invalid request:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours":6,"attendance_percentage":120}'
```

## Exercise 3: Streamlit interface

```powershell
py -m streamlit run streamlit_app.py
```

Enter a student name, study hours, and attendance percentage, then select **Predict**. The app explains that it loads the saved model and uses both numeric inputs. The chart shows the predicted class across study-hour values while holding the entered attendance fixed, and highlights the entered study hours.
