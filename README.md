# Student Pass Predictor

## Setup

From PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
```

The training script fits a classifier from example study-hour records, saves it as `student_pass_model.joblib`, reloads that file, and predicts results for three different study-hour values. The saved file contains the learned model parameters, so new predictions can be made by loading the file; fitting is only needed again when changing the training data or model.

## Student website

```powershell
python -m streamlit run streamlit_app.py
```

## Flask API

In a second terminal:

```powershell
python app.py
```

Test the endpoint with `curl.exe`:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours": 6, "attendance_percentage": 92}'
```

Attendance must be between 0 and 100. It is returned in the response but does not affect the prediction because the trained model uses study hours as its only feature. To make attendance influence predictions, include it in the training data and model features, then retrain the model.

## Tests

```powershell
python -m unittest discover -s tests -v
```