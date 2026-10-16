# How to Run

## Setup

```powershell
cd C:\Users\ACER\Exercise
pip install scikit-learn joblib flask streamlit pandas numpy
```

## Exercise 1

```powershell
python train_model.py
```

## Exercise 2

```powershell
python app.py
```

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"study_hours":4.5,"attendance_percentage":80}'
```

## Exercise 3

```powershell
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.
