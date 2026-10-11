# Student Prediction Exercises

This folder contains the completed model-persistence, Flask API, and Streamlit exercises.

## Run in VS Code

Open this folder in VS Code, then run these commands in its terminal.

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py
```

Training saves `student_pass_model.joblib`, reloads it, and prints three sample predictions. The saved model is also included in this folder.

## Run the Flask API

```bash
python app.py
```

Test it in another terminal:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 5, "attendance_percentage": 80}'
```

The endpoint validates study hours and attendance, then returns the inputs and prediction as JSON.

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app accepts a student name, study hours, and attendance, then shows the prediction, explanation, and chart. To open it on a phone on the same Wi-Fi, run:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

Open `http://<computer-local-IP>:8501` on the phone.

The apps load the saved model for predictions. They do not retrain it for every student.
