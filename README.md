# Student Prediction Exercises — Alternative Code

This is an alternative implementation of Exercises 1–3. It uses a different code structure while keeping the same model type, sample training data, prediction labels, and API/interface behavior as the original package.

## Install
```bash
pip install -r requirements.txt
```

## Exercise 1
```bash
python train_model.py
```
This trains and saves `student_pass_model.joblib`, reloads it, then prints three predictions.

## Exercise 2: Flask API
```bash
python app.py
```
POST JSON to `http://127.0.0.1:5000/predict`:
```json
{
  "study_hours": 5,
  "attendance_percentage": 80
}
```
Test using Postman or:
```bash
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":80}"
```

## Exercise 3: Streamlit
In a second terminal:
```bash
streamlit run streamlit_app.py
```

The model is trained using both study hours and attendance percentage. It is loaded from disk for each prediction and is not retrained on every request. The generated training data is for classroom demonstration only.
