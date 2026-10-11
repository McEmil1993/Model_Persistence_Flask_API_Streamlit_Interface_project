# Student pass prediction exercise

## Install and train

```powershell
python -m pip install -r requirements.txt
python train_model.py
```

`train_model.py` saves `student_pass_model.joblib`, reloads it with `joblib.load()`,
and prints three example predictions. The bundled data is illustrative because the
starter folder did not contain a dataset. Replace it with representative, labeled
student data for a meaningful model.

## Flask API

Start the API after training:

```powershell
python app.py
```

In another PowerShell window, test it with `curl.exe` (PowerShell has its own
`curl` alias, so use the executable name):

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours":6,"attendance_percentage":85}'
```

The endpoint validates attendance as a number from 0 through 100. The current
model was trained with study hours alone, so attendance is included in the
response but does not affect the predicted class. To use attendance in predictions,
add it as a training feature and retrain the model.

## Streamlit interface

```powershell
streamlit run streamlit_app.py
```

The interface asks for the student's name, study hours, and attendance percentage,
then displays the predicted class, a short explanation, attendance for reference,
and a chart comparing study hours with the predicted class. Attendance must be
between 0 and 100 but does not affect the current model's prediction.
