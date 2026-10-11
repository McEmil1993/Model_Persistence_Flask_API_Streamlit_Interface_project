STUDENT PASS PREDICTION — ACTIVITY STARTER SOLUTION
==================================================

Included files
- train_model.py: trains the example model, saves it as student_pass_model.joblib,
  loads it, and prints three predictions.
- app.py: Flask API with POST /predict. Accepts study_hours and
  attendance_percentage, validates attendance from 0 to 100, and returns JSON.
- streamlit_app.py: simple interface with student name, study hours, Predict button,
  result/explanation, and a bonus chart.
- requirements.txt: Python packages needed.

IMPORTANT
The provided exercise says the model uses study hours only. Therefore, attendance
is validated and returned by the API but does NOT affect the prediction. To make
attendance influence predictions, you would need training data with attendance
and retrain a model using both features.

These files use a small example dataset because the uploaded lesson/exercise files
contain instructions but do not include train_model.py, app.py, streamlit_app.py,
or the original training dataset. If your instructor provided those starter files,
use them as the base and transfer the requested changes into them.

SETUP
1. Install Python 3.10 or newer.
2. Open a terminal/Command Prompt in this folder.
3. Install packages:
   pip install -r requirements.txt
4. Train and save the model:
   python train_model.py

RUN THE STREAMLIT APP
   streamlit run streamlit_app.py

RUN THE FLASK API
   python app.py

TEST THE API WITH curl.exe (Windows PowerShell)
   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"

Expected response format (the predicted class may depend on the model):
{
  "study_hours": 5.0,
  "attendance_percentage": 90.0,
  "prediction": "Pass",
  "note": "Prediction uses study_hours only; attendance is not a model feature."
}

INVALID ATTENDANCE TEST
   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":120}"

This should return HTTP 400 and an error message because attendance must be between 0 and 100.

ANSWER FOR EXERCISE 1 EXPLANATION
The model does not need to be trained again because joblib.dump() saves the already
trained model to a file, and joblib.load() restores that trained model. The loaded
model can make predictions on new study-hour values using patterns learned earlier.