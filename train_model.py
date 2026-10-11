
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("student_data.csv")

print("Training the student performance model...")


X = data[["study_hours"]]
y = data["result"]


model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print("Model training completed successfully.")


joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")


loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully.")


test_hours = [2, 5, 8]

print("\nSample Predictions:")

for hours in test_hours:
    sample = pd.DataFrame({"study_hours": [hours]})
    prediction = loaded_model.predict(sample)[0]

    print(f"Study Hours: {hours} -> Predicted Result: {prediction}")

print("\nPredictions completed successfully.")


print(
    "\nExplanation: The trained model is saved in "
    "student_pass_model.joblib. We can load the saved model "
    "and make predictions without training it again because "
    "the file contains the trained model. Retraining is needed "
    "when we want to update the model using new or changed "
    "training data."
)
