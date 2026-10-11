STUDENT PASS PREDICTION — 3 EXERCISES
======================================

FILES
-----
train_model.py       Exercise 1: trains, saves, loads, and tests the model
app.py               Exercise 2: Flask JSON API
streamlit_app.py     Exercise 3: Streamlit interface
students.csv         Small example dataset
requirements.txt     Required Python packages

IMPORTANT
---------
The included students.csv is sample/demo data created for this project.
The predictions are for learning and are not guaranteed to reflect real student outcomes.

SETUP IN VS CODE (Windows)
--------------------------
1. Install Python 3.10 or newer, then install VS Code.
2. Extract this ZIP and open the student_pass_ml_project folder in VS Code.
3. Open Terminal > New Terminal.
4. (Recommended) Create and activate a virtual environment:

   py -m venv .venv
   .venv\Scripts\activate

   If PowerShell blocks activation, use Command Prompt in VS Code, or run:
   .venv\Scripts\python.exe -m pip install -r requirements.txt

5. Install packages:

   python -m pip install -r requirements.txt

6. Train the model and create student_pass_model.joblib:

   python train_model.py

   This prints the test accuracy and three predictions. The .joblib file is generated
   automatically in the project folder.

EXERCISE 1 — MODEL PERSISTENCE
------------------------------
Run:
   python train_model.py

joblib.dump() saves the trained model to student_pass_model.joblib.
joblib.load() loads the saved model so it can make predictions without retraining.
The sample script predicts outcomes for 2, 5, and 8 study hours with Results may depend on the training data and model.

EXERCISE 2 — FLASK API
----------------------
First make sure student_pass_model.joblib exists, then run:
   python app.py

Keep this terminal open. In a second VS Code terminal, test with curl.exe:

   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6}"

Expected response format (the prediction and confidence will vary):
{
  "study_hours": 6.0,
  "prediction": "PASS",
  "predicted_class": 1,
  "confidence_for_predicted_class": 0.8
}

Validation tests:
- Negative study_hours returns an error.
- Missing study_hours returns an error.
- Non-numeric study_hours returns an error.

To test with Postman, use POST, URL http://127.0.0.1:5000/predict,
Body > raw > JSON, then enter:
{
  "study_hours": 6
}

EXERCISE 3 — STREAMLIT
----------------------
Stop Flask with Ctrl+C if desired. Run:
   streamlit run streamlit_app.py

The app should open in your browser. Enter a student name and study hours,
then click Predict result. It displays the prediction and a simple chart.

TROUBLESHOOTING
---------------
- If 'python' is not recognized, install Python and select it in VS Code:
  Ctrl+Shift+P > Python: Select Interpreter.
- If a module is missing, run: python -m pip install -r requirements.txt
- If the model file is missing, run: python train_model.py
- Run commands from the project folder (the folder containing these files).