# Student Pass Prediction (Termux, no scikit-learn)

## Setup
    pkg update && pkg upgrade -y
    pkg install python python-pip python-numpy unzip curl -y
    pip install joblib flask

## Exercise 1
    python train_model.py

## Exercise 2 (Flask API) - run, then test from a second Termux session
    python app.py
    curl "http://127.0.0.1:5000/predict?study_hours=6&attendance_percentage=85"
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours":2,"attendance_percentage":50}'
    curl "http://127.0.0.1:5000/predict?study_hours=5&attendance_percentage=150"

## Exercise 3 (web interface)
    python web_app.py        # open http://127.0.0.1:8501 in your phone browser
(streamlit_app.py is included too, if Streamlit installs on your device.)
