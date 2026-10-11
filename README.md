# Student Pass Prediction Demo

This workspace is a runnable example using a small, illustrative dataset. The model predicts pass/fail from study hours only. Attendance is validated and returned by the API, but does not affect the prediction. Replace the example training data in `train_model.py` with real data before using it for meaningful predictions.

## Install and train

```powershell
pip install -r requirements.txt
python train_model.py
```

The training script saves `student_pass_model.joblib`, loads it again, and prints predictions for three study-hour values. Inference reuses the learned parameters in that file; it does not need to fit the model again unless the data or model changes.

## Run the API

```powershell
python app.py
```

Test with `curl.exe` in a second PowerShell window:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data-raw '{"study_hours":6,"attendance_percentage":92}'
```

The endpoint accepts `study_hours` from 0 to 24 and `attendance_percentage` from 0 to 100. It returns both inputs and the predicted class and label as JSON. In Postman, send a `POST` request to `http://127.0.0.1:5000/predict` with a raw JSON body containing those two fields.

## Run the Streamlit interface

```powershell
streamlit run streamlit_app.py
```