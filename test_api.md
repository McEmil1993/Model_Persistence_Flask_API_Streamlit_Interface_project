# Testing the Flask endpoint

Start the server: `python app.py`

## curl.exe (Windows PowerShell)
```
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 85}"
```

## curl (Mac/Linux/Git Bash)
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours": 6, "attendance_percentage": 85}'
```

## Invalid attendance (should return HTTP 400)
```
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 120}"
```

## Postman
POST http://127.0.0.1:5000/predict, Body > raw > JSON:
{"study_hours": 6, "attendance_percentage": 85}
