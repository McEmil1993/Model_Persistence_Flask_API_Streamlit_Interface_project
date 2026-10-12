# Student Pass Prediction Activity

This example implements the model-persistence, Flask API, and Streamlit
exercises. The small training dataset in [train_model.py](./train_model.py) is
illustrative only; it is not real student data and should not be used to make
actual academic decisions.

## Setup on Windows

Open PowerShell in the project folder:

```powershell
cd C:\MY_AllFolder\Boiser
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the `py` launcher is unavailable, use `python` in its place. Install the
project dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Exercise 1: save and reuse a model

Train the model, save it, reload it, and make three sample predictions:

```powershell
python train_model.py
```

The script learns a study-hours decision threshold, saves it as
`student_pass_model.joblib` with `joblib.dump()`, then reloads it with
`joblib.load()` and predicts results for 2, 5, and 8 study hours. Training
learns the model's decision rule; the saved file preserves that rule, so
predictions can be made by loading the model without fitting it again.

## Exercise 2: Flask API

After training, start the API in one PowerShell window:

```powershell
python app.py
```

The API listens at `http://127.0.0.1:5000`. In a second PowerShell window, test
the endpoint with `curl.exe`:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"study_hours":6,"attendance_percentage":92}'
```

The JSON response includes both inputs and the predicted class. Attendance is
validated from 0 through 100, but it is not a model feature, so changing
attendance alone does not affect the prediction. Try an invalid attendance
value to see the HTTP 400 validation error:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"study_hours":6,"attendance_percentage":120}'
```

## Exercise 3: Streamlit interface

In another PowerShell window, run:

```powershell
python -m streamlit run streamlit_app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`. Enter a
name and study hours, then select **Generate prediction**. The page shows the
result, explains that the saved model uses study hours only, and plots predicted
class (0 = Fail, 1 = Pass) against study hours.

## Using attendance in predictions

The API accepts and returns attendance for the exercise, but the provided model
is trained on study hours only. To make attendance affect predictions, include
attendance as a feature in a suitable training dataset, retrain the model, and
update the prediction input to provide both features. Do not treat the
illustrative data in this example as a reliable predictor.
  