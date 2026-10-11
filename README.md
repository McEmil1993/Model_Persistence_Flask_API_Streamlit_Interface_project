# Student Pass Prediction Activity

This example implements the model-persistence, Flask API, and Streamlit
exercises. The small training dataset in `train_model.py` is illustrative only;
it is not real student data and should not be used to make actual academic
decisions.

## Setup

In PowerShell, from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Exercise 1: save and reuse a model

Run:

```powershell
python train_model.py
```

The script learns a study-hours decision threshold, saves it as
`student_pass_model.joblib` with `joblib.dump()`, then reloads it with
`joblib.load()` and predicts results for 2, 5, and 8 study hours. Training
learns the model's decision rule; the saved file preserves that rule, so
predictions can be made by loading the model without fitting it again.

## Exercise 2: Flask API

Start the API after training the model:

```powershell
python app.py
```

In another PowerShell window, test a valid request with `curl.exe` (piping the
JSON body avoids PowerShell changing the quotes):

```powershell
'{"study_hours":6,"attendance_percentage":92}' | curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data-binary "@-"
```

The JSON response includes both inputs and the predicted class. Attendance is
validated from 0 through 100, but it is not a model feature, so changing
attendance alone does not affect the prediction. For example, try an invalid
attendance value:

```powershell
'{"study_hours":6,"attendance_percentage":120}' | curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data-binary "@-"
```

The invalid request returns HTTP 400 with a validation error.

## Exercise 3: Streamlit interface

Run:

```powershell
streamlit run streamlit_app.py
```

Enter a name and study hours, then select **Predict**. The page shows the
predicted result, explains that the saved model uses study hours only, and
plots predicted class (0 = Fail, 1 = Pass) against study hours.

## Using attendance in predictions

The API accepts and returns attendance for the exercise, but the provided
model is trained on study hours only. To make attendance affect predictions,
include attendance as a feature in a real training dataset, retrain the model,
and update the prediction input to provide both features. Do not treat the
illustrative data in this example as a reliable predictor.





python app.py


cd /c/MY_AllFolder/Python/Activity
source .venv/Scripts/activate
python train_model.py



cd /c/MY_AllFolder/Python/Activity
source .venv/Scripts/activate



curl.exe -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours":6,"attendance_percentage":92}'


  python -m streamlit run streamlit_app.py
  