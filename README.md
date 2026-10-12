# Student Pass Prediction Exercises

This example trains a model using study hours, saves it as `student_pass_model.joblib`, and reuses it in a Flask API and Streamlit interface. The sample training data is included in `train_model.py`; replace it with your course dataset when available.

## Install and train

```powershell
python -m pip install -r requirements.txt
python train_model.py
```

The model only needs to be trained when its training data or model settings change. Predictions load the saved model and run inference, so they do not repeat the training step.

## Flask API

```powershell
python app.py
```

In another terminal, send a request with `curl.exe`:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":6,\"attendance_percentage\":92}"
```

The response includes the supplied study hours and attendance percentage plus the predicted class and a readable result. Attendance is validated and returned, but it does not affect the prediction because the trained model currently uses study hours alone. To have attendance influence predictions, add it to the training features in `train_model.py`, retrain, and then update the prediction calls to pass both features.

## Streamlit interface

```powershell
streamlit run streamlit_app.py
```

Enter a student name and study hours, then select **Predict**. The interface displays the result, explains that it uses a saved model, and plots the predicted class over study hours.
