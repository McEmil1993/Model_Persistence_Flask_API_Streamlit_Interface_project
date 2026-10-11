
import joblib

model = joblib.load("student_pass_model.joblib")

print("Model loaded successfully!")

prediction = model.predict([[5]])

if prediction[0] == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")


# Explanation:
# The model does not need to be trained again because it was
# already trained and saved using joblib.dump().
# joblib.load() loads the saved model and allows it to make
# predictions without repeating the training process.
# Retraining is only necessary when updating the model with
# new training data or additional features.