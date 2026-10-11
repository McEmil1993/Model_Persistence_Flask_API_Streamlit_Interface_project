# How to Run

## Setup

```powershell
cd C:\Users\ACER\James_Exercise
pip install scikit-learn joblib flask streamlit pandas numpy altair
```

## Exercise 1 — Train the model

```powershell
python train_model.py
```

Possible output:

```text
Training accuracy: 0.952
Test accuracy: 0.922
Model saved to student_pass_model.joblib
Model loaded from student_pass_model.joblib
Study hours=2.0, Attendance=60.0% -> FAIL (P(pass)=0.41)
Study hours=5.0, Attendance=75.0% -> PASS (P(pass)=0.99)
Study hours=8.0, Attendance=90.0% -> PASS (P(pass)=1.00)

Why no retraining is needed: joblib.dump() persists the fitted pipeline (scaler statistics plus classifier coefficients/intercept). joblib.load() restores that exact state, so predict()/predict_proba() work immediately without calling fit() again.
```

## Exercise 2 — Flask API

Terminal 1 (server, blocks):

```powershell
python app.py
```

Possible output:

```text
* Serving Flask app 'app'
* Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

Terminal 2 (tests):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/ -Method Get
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"study_hours":5,"attendance_percentage":75}'
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"study_hours":5,"attendance_percentage":150}'
```

Possible outputs:

```text
# GET /
message : Student pass API. POST JSON to /predict.

# POST valid
study_hours           : 5
attendance_percentage : 75
prediction            : 1
result                : PASS
probability_pass      : 0.9865

# POST invalid attendance -> 400
{"error":"'attendance_percentage' must be between 0 and 100."}
```

## Exercise 3 — Streamlit UI

```powershell
streamlit run streamlit_app.py
```

Possible output:

```text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Then in the browser:

- Fill `Student name`, `Study hours per week`, `Attendance (%)`, click `Run prediction`.
- Example `Sam / 4.5h / 80%` -> `Metric: PASS`, `P(pass) = 0.98`, doughnut `Pass chance ~0.98 vs Fail chance ~0.02`.
- Bottom doughnut at `80%` -> `Across 0-12h: 23 PASS vs 2 FAIL`.
