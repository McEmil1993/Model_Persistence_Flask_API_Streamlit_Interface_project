import joblib
from sklearn.linear_model import LogisticRegression

# Study hours -> 0 = Fail, 1 = Pass
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")

# Load the saved model (no retraining)
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded!")

# Three predictions using different study hours
for hours in [2, 5.5, 9]:
    result = loaded_model.predict([[hours]])[0]
    print(f"Study hours: {hours} -> {'Pass' if result == 1 else 'Fail'}")

# Explanation: Why the model does not need to be trained again
# Everything the model learned during fit() (its coefficients and intercept)
# is stored inside the saved .joblib file. joblib.load() restores the same
# trained model, so making predictions only applies those learned values to
# the new input. Training is done once, and the saved model is reused.
print()
print("Explanation:")
print("The model does not need to be trained again because everything it learned")
print("during fit() (its coefficients and intercept) is stored inside the saved")
print(".joblib file. joblib.load() restores the same trained model, so making")
print("predictions only applies those learned values to the new input.")