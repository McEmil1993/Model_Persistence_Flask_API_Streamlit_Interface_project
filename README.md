# Student Pass Predictor

## Run order

```bash
python train_model.py
python train_model_v2.py
python app.py                      # Flask API
streamlit run streamlit_app.py     # Streamlit UI (separate terminal)
```

## Exercise 1: Model Persistence

Modify train_model.py to save the trained model using a different filename, such as student_pass_model.joblib.

Tasks:

- Save the model using joblib.dump().
- Load the model using joblib.load().
- Make three predictions using different study hours.
- Explain why the model does not need to be trained again to make those predictions.

## Exercise 2: Flask API

Modify app.py to accept a second input named attendance_percentage.

Tasks:

- Accept both study hours and attendance percentage.
- Validate that attendance is between 0 and 100.
- Return both values and the model prediction in JSON.
- Test the endpoint using Postman or curl.exe.

Note: The provided model currently uses only study hours. Students must also update and retrain the model if they want attendance to influence its predictions.

## Exercise 3: Streamlit Interface

Improve streamlit_app.py by adding:

- A student name input.
- A study-hours input.
- A prediction button.
- A result message.
- A short explanation of how the prediction was generated.

Bonus: Add a chart that compares the study hours with the predicted class.
