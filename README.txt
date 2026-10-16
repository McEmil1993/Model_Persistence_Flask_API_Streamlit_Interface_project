STUDENT PASS PREDICTION - VISUAL STUDIO CODE PROJECT

FILES
- train_model.py       Trains and saves the model, then reloads it and makes 3 predictions.
- student_pass_model.joblib  Created automatically after training.
- app.py               Flask API.
- streamlit_app.py     Streamlit interface and bonus chart.
- requirements.txt     Python packages needed.

BEFORE YOU START
1. Install Python 3 if it is not installed.
2. Open this folder in Visual Studio Code.
3. Open Terminal > New Terminal.
4. (Recommended) Create and activate a virtual environment:

   Windows:
   py -m venv .venv
   .venv\Scripts\activate

5. Install packages:

   py -m pip install -r requirements.txt

EXERCISE 1: MODEL PERSISTENCE
Run:
   py train_model.py

This creates student_pass_model.joblib and prints three predictions for 2, 5, and 8 study hours.
The example dataset is for demonstration. Use your instructor's dataset if one was provided.

EXERCISE 2: FLASK API
First make sure you have run train_model.py.
Start the API:
   py app.py

Open this address in your browser to check the API:
   http://127.0.0.1:5000/

Test with curl.exe in a SECOND terminal while Flask is running:
   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 5, \"attendance_percentage\": 85}"

To test invalid attendance:
   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 5, \"attendance_percentage\": 120}"

The API currently predicts from study_hours only. Attendance is validated and returned but does not affect the result, matching the exercise note.

EXERCISE 3: STREAMLIT
Stop Flask with Ctrl+C, or leave it running and open another terminal.
Run:
   py -m streamlit run streamlit_app.py

Streamlit will show a local URL in the terminal, usually:
   http://localhost:8501

NOTES
- Internet is needed to install packages for the first time, but running the project after installation can generally be done offline.
- The model is a simple classroom demonstration, not a reliable real-world assessment tool.
