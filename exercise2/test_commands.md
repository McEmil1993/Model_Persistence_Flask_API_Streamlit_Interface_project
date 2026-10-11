# Testing the endpoint

Run first:

    python train_model.py
    python app.py

## curl.exe (Windows PowerShell)

Valid request (GET):

    curl.exe "http://127.0.0.1:5000/predict?study_hours=5&attendance_percentage=85"

Valid request (POST JSON):

    curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 5, \"attendance_percentage\": 85}"

Invalid attendance (expect HTTP 400):

    curl.exe "http://127.0.0.1:5000/predict?study_hours=5&attendance_percentage=150"

Missing input (expect HTTP 400):

    curl.exe "http://127.0.0.1:5000/predict?study_hours=5"

## Postman

Method GET, URL `http://127.0.0.1:5000/predict`, add Params
`study_hours` = 5 and `attendance_percentage` = 85, then Send.