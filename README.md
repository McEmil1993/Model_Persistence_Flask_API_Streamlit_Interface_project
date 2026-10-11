# Student Pass Prediction Project

**Prepared for:** Michael Felisilda Canja

A small educational project for the three activities in `Exercise(4).html`:
1. Model persistence using `joblib`
2. Flask prediction API with study hours and attendance validation
3. Streamlit user interface with student name, study hours, attendance, prediction, explanation, and chart

## Folder structure

```text
Michael_Canja_Student_Pass_Prediction/
├── app.py
├── train_model.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── student_data.csv       # created by train_model.py
└── models/
    └── student_pass_model.joblib  # created by train_model.py
```

## 1. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Train and save the model

```powershell
python train_model.py
```

This creates `data/student_data.csv` and saves the trained model as `models/student_pass_model.joblib`. It also reloads the model and prints three example predictions.

## 3. Run the Streamlit interface

```powershell
streamlit run streamlit_app.py
```

If `streamlit` is not recognized, use:

```powershell
python -m streamlit run streamlit_app.py
```

## 4. Run the Flask API

Open a separate terminal in the project folder:

```powershell
python app.py
```

The API runs at `http://127.0.0.1:5000`.

### Test with curl.exe (another terminal)

```powershell
curl.exe -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" --data-binary "{\"student_name\":\"Michael Felisilda Canja\",\"study_hours\":4,\"attendance_percentage\":90}"
```

Attendance must be between 0 and 100. Study hours must be between 0 and 24.

**Important:** This example model uses only `study_hours`. The API validates and returns `attendance_percentage`, but attendance does not affect the prediction unless you update the training dataset and train the model with both features.

## 5. Git commands

Configure Git identity once if needed:

```powershell
git config --global user.name "Michael Felisilda Canja"
git config --global user.email "YOUR_EMAIL_ADDRESS"
```

Initialize and commit:

```powershell
git init
git add .
git commit -m "Michael Felisilda Canja"
git branch -M michael-canja
```

Add your repository remote only after confirming the repository URL and that you have permission to push:

```powershell
git remote add origin YOUR_REPOSITORY_URL
git push -u origin michael-canja
```

Do not put a GitHub personal access token directly in a remote URL or commit it to a file. If a token was previously exposed, revoke it and create a new one.
