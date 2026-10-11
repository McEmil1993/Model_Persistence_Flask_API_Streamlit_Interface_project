# Model Persistence and Basic Deployment Exercise

This project follows the three exercises from the uploaded lesson:

1. Model Persistence
2. Flask API
3. Streamlit Interface

IMPORTANT:
The uploaded Exercise.html does not contain the original training dataset.
Therefore, `train_model.py` uses a small sample dataset so the project can run
immediately. If your instructor gave you an original model/dataset, replace the
sample training part with that model/dataset.

## 1. Install the required packages

Open VS Code Terminal / PowerShell inside this folder:

    py -m pip install -r requirements.txt

## 2. Train and save the model

Run:

    python train_model.py

This creates:

    student_pass_model.joblib

The script also makes three sample predictions.

## 3. Run Flask

Run:

    python app.py

Open:

    http://127.0.0.1:5000

### Test Flask with curl

In another PowerShell terminal:

    curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 5, \"attendance_percentage\": 90}"

The attendance value is validated from 0 to 100.

The current sample model uses study hours only, exactly matching the note in
the uploaded exercise: attendance must be included in the request/response,
but the model must be retrained with attendance if attendance should affect
the prediction.

## 4. Run Streamlit

Run:

    streamlit run streamlit_app.py

The Streamlit app includes:
- Student Name input
- Study Hours input
- Decimal values such as 10.09
- Predict button
- Prediction result
- Explanation
- Graph shown only after clicking Predict
- A visible diamond point for the entered study-hours value

## Files

- `train_model.py` - trains, saves, loads, and tests the model
- `app.py` - Flask API
- `streamlit_app.py` - Streamlit interface
- `requirements.txt` - required Python packages