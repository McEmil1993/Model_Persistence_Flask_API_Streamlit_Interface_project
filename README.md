# Student Pass Prediction Exercises

## Setup

```powershell
py -m pip install -r requirements.txt
```

## Exercise 1: persistence

```powershell
py train_model.py
```

This trains the example model, saves it as `student_pass_model.joblib`, loads it again, and makes three predictions. A saved model already contains learned parameters, so it can predict without retraining. The project uses a lightweight pure-Python classifier, avoiding native extensions that can be blocked by Windows Application Control.

## Exercise 2: Flask API

```powershell
py app.py
```

Test it in a second terminal:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 85}"
```

`attendance_percentage` must be from 0 through 100. It is returned in the response, but the supplied model uses only study hours.

## Exercise 3: Streamlit UI

```powershell
py -m streamlit run streamlit_app.py
```

The UI accepts an attendance percentage from 0 through 100 and displays it with
the prediction. Attendance is recorded for context; the example model currently
uses study hours only.
