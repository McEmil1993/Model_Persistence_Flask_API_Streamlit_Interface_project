
# Student Pass Prediction System

**Author:** Jean Chawe Cabañero

## Project Description

The Student Pass Prediction System is a Python-based machine learning project that predicts whether a student will pass or fail based on their study hours.

This project demonstrates model persistence, API development using Flask, and a user interface using Streamlit.

## Technologies Used

- Python
- Scikit-learn
- Joblib
- Flask
- Streamlit
- Pandas

## Project Files

- `train_model.py` — Trains the machine learning model, saves it, loads it, and makes three predictions.
- `app.py` — Provides a Flask API that accepts study hours and attendance percentage and returns a prediction in JSON format.
- `streamlit_app.py` — Provides an interactive interface for entering a student's name and study hours and viewing the predicted result.
- `student_pass_model.joblib` — Stores the trained machine learning model.

## Installation

Install the required Python libraries using the terminal:

```bash
python -m pip install scikit-learn joblib flask streamlit pandas
```

## How to Run

### Exercise 1: Model Persistence

Run the following command:

```bash
python train_model.py
```

The program trains the model, saves it as `student_pass_model.joblib`, loads the saved model, and makes three predictions using different study hours.

### Exercise 2: Flask API

Run the Flask application:

```bash
python app.py
```

Send a POST request to:

`http://127.0.0.1:5000/predict`

Example JSON request:

```json
{
  "study_hours": 6,
  "attendance_percentage": 90
}
```

The API validates the attendance percentage and returns the input values and predicted result in JSON format.

### Exercise 3: Streamlit Interface

Run the Streamlit application:

```bash
python -m streamlit run streamlit_app.py
```

The interface allows users to enter a student name, input study hours, click the prediction button, and view the result, explanation, and chart.

## Important Note

The current model uses study hours only to predict whether a student will pass or fail. Attendance percentage is accepted and validated by the Flask API, but it does not affect the prediction.

To include attendance in the prediction, the model must be retrained using training data containing both study hours and attendance percentage.

The predictions are for educational demonstration purposes and do not guarantee actual academic performance.

## Author

**Jean Chawe Cabañero**
