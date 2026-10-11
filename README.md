# Student Pass Prediction Exercises

This project contains the code for all three exercises:

1. Model persistence with `joblib`
2. Flask API with study hours and attendance validation
3. Streamlit UI for student prediction

## 1) Train and save the model

Run:

```bash
python train_model.py
```

This creates `student_pass_model.joblib` in the project folder.

## 2) Run the Flask API

Run:

```bash
python app.py
```

Then test with:

```bash
curl.exe "http://127.0.0.1:5000/predict?study_hours=3&attendance_percentage=80"
```

or POST JSON:

```bash
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 3, \"attendance_percentage\": 80}"
```

## 3) Run the Streamlit app

Run:

```bash
"C:\Users\ACER\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\streamlit.exe" run streamlit_app.py --server.headless true --server.port 8501
```

Then open:

```text
http://localhost:8501
```

## Notes

- The saved model is loaded using `joblib.load()` and reused for predictions.
- The API validates attendance between 0 and 100.
- The model uses both study hours and attendance percentage. Run `python train_model.py` to retrain and save it.
- Training uses synthetic classroom examples: scores are generated as `10 + 5 * study_hours + 0.4 * attendance_percentage`. This demonstrates both inputs; it is not validated on real student outcomes.
- At 5 study hours, attendance of 40%, 80%, and 100% produces scores of 51, 67, and 75 respectively.
- The Streamlit form displays attendance, and the chart holds attendance at the submitted percentage. Scores can fall outside 0?100.
- Restart the API and Streamlit after retraining so they load the updated saved model.
