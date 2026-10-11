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

`attendance_percentage` must be from 0 through 100. It now affects the prediction: attendance adjusts the effective study-hours score around a 75% reference point. The response includes the predicted class and estimated pass probability.

## Exercise 3: Streamlit UI

Enter study hours and attendance from 0 through 100. Higher attendance increases the effective score and lower attendance reduces it.

```powershell
py -m streamlit run streamlit_app.py
```
