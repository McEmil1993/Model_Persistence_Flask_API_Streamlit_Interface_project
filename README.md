# Student Pass Prediction Exercise

## 1. Train and persist the model

```powershell
pip install -r requirements.txt
python train_model.py
```

This creates `student_pass_model.joblib`, reloads it, and makes three predictions for 2, 5, and 8 study hours. The saved file contains the fitted model, so the model does not need training again before making predictions.

## 2. Run the Flask API

```powershell
python app.py
```

Test it in another PowerShell terminal:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 88}"
```

Attendance must be from 0 to 100. It is returned in the response, but does not yet affect the prediction because the training data currently has only study hours.

## 3. Run the Streamlit interface

```powershell
streamlit run streamlit_app.py
```
