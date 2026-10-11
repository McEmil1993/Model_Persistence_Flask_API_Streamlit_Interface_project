# Run Flow: Student Pass Prediction

Run all commands from the `Model` folder in Git Bash with `(.venv)` active.

## 0. Setup (once)

```bash
cd "/c/Users/felja/OneDrive/Desktop/4TH YEAR_FILES/Sir-Emil/ELEC 3/Jhustine Vallente/Model"
source .venv/Scripts/activate
python -m pip install joblib scikit-learn pandas numpy flask streamlit matplotlib
```

If the prompt already shows `(.venv)` and you are in the `Model` folder, skip the `cd` and `source` lines.





## Exercise 1: Model Persistence TERMINAL IPA RUN

```bash
python train_model.py -- KANI UNA/FIRST OH
python predict_saved_model.py -- IKADUHA/SECOND OH
```

- `train_model.py` trains the model and creates `student_pass_model.joblib`.
- `predict_saved_model.py` loads the saved model and prints three predictions (2, 5, and 9 study hours).

## Exercise 2: Flask API

```bash
python train_model_v2.py
python app.py

TAMAN DIRI
```

- `train_model_v2.py` retrains with study hours and attendance, and creates `student_pass_model_v2.joblib`.
- `app.py` starts the server at `http://127.0.0.1:5000`. **Leave this terminal open.**

Open a **second** terminal (activate `.venv` there too if needed) and test the endpoint:



KARI SA GITBASH NI!! 
```bash
# Valid request
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours": 6, "attendance_percentage": 85}'

# Invalid attendance (expect a 400 error)
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours": 6, "attendance_percentage": 120}'

# Missing field (expect a 400 error)
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours": 6}'
```




In Git Bash, plain `curl` works with single quotes around the JSON. In PowerShell, use `curl.exe` with escaped double quotes instead.

Stop the Flask server with `Ctrl + C` when done.

## Exercise 3: Streamlit Interface

```bash
python -m streamlit run streamlit_app.py ''' KANI LAST
```










- Opens at `http://localhost:8501`.
- Stop it with `Ctrl + C`.

## Notes

- Run `python train_model.py` (Exercise 1) before Exercise 3, because the Streamlit app needs `student_pass_model.joblib`.
- Flask (port 5000) and Streamlit (port 8501) are independent, so only one needs to be running at a time.
- Use `python -m pip` and `python -m streamlit` so everything runs in the same `.venv` Python.
- If `python` is not recognized in Git Bash, try `py` instead.
