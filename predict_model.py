import joblib

model = joblib.load("student_pass_model.joblib")

hours = [[2], [5], [8]]
predictions = model.predict(hours)

for hour, prediction in zip(hours, predictions):
    print(f"Study hours: {hour[0]}, Prediction: {prediction}")