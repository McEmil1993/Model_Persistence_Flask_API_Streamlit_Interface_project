# Student Pass Predictor

## Setup
    pip install -r requirements.txt
    python train_model.py              # Exercise 1 (saves student_pass_model.joblib)
    python train_model_attendance.py   # Exercise 2 (hours + attendance model)

## Exercise 2 - run and test the API
    python app.py

curl.exe (Windows):
    curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 85}"

Invalid attendance (returns 400):
    curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 5, \"attendance_percentage\": 120}"

## Exercise 3 - Streamlit
    streamlit run streamlit_app.py
