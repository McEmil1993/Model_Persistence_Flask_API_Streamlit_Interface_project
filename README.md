# Student Pass Predictor

This project demonstrates model persistence, a Flask JSON API, and a Streamlit interface. The example model uses **study hours only**. The API validates and returns attendance but does not use it to calculate the prediction.

## Set up on Windows

From this folder in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
```

`train_model.py` trains a logistic-regression classifier, saves it as `student_pass_model.joblib`, reloads it with `joblib.load()`, and makes three sample predictions. The API and Streamlit app reuse that saved model, so they do not retrain it for each prediction.

## Test with Postman

1. Start the API in a PowerShell terminal: `python app.py`.
2. In Postman, choose **Import** and import `Student Pass Predictor.postman_collection.json`.
3. Run **Predict pass result** for a successful prediction, or **Reject attendance above 100** to confirm validation.

The successful request is `POST http://127.0.0.1:5000/predict` with JSON body:

```json
{
  "study_hours": 6,
  "attendance_percentage": 85
}
```

The endpoint returns both input values, the predicted class (`0` or `1`), and the result (`Fail` or `Pass`). Attendance must be from 0 through 100. Study hours must be from 0 through 24.

Equivalent PowerShell test using curl.exe:

```powershell
'{"study_hours":6,"attendance_percentage":85}' | curl.exe -H "Content-Type: application/json" --data-binary '@-' http://127.0.0.1:5000/predict
```

## Run the Streamlit interface

In another PowerShell terminal:

```powershell
streamlit run streamlit_app.py
```

The interface accepts a student name and study hours, displays the prediction and its explanation, and charts the classifier's predicted class across the study-hours range.

To make attendance influence predictions, add it as a model feature in the training data, train on both values, and retrain the model. Merely sending attendance to the current endpoint does not change the result.