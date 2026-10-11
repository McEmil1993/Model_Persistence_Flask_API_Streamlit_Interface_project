# Student outcome predictor

This version uses **study hours and attendance percentage** as model features. The small in-script training dataset is illustrative; replace it with a representative student dataset before using predictions for real decisions.

## Train and save

```powershell
python -m pip install -r requirements.txt
python train_model.py
```

Training writes `student_pass_model.joblib`. The training script immediately reloads that saved file and demonstrates three predictions. Persistence means the fitted decision rules are stored on disk; inference can load and reuse them without repeating training. Retrain only when the training data or model design should change. Attendance was included as a feature here, so it can influence the output.

## Flask API

```powershell
python app.py
```

Send this JSON to `http://127.0.0.1:5000/predict` with Postman or curl.exe:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours":5,"attendance_percentage":82}'
```

The response includes both inputs and the predicted class. Attendance outside 0–100 or missing/invalid inputs return HTTP 400.

## Streamlit app

```powershell
streamlit run streamlit_app.py
```

Enter a name, study hours, and attendance, then submit. The app shows the result, estimated pass probability, a short model explanation, and a chart showing the predicted class across attendance levels while holding study hours fixed.
