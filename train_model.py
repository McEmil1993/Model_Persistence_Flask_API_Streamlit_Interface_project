"""Train, save, load, and use a simple student-pass prediction model."""

import joblib
from sklearn.linear_model import LogisticRegression


# Step 1: Make a small, fictional dataset.
# Each inner list is [study_hours, attendance_percentage] for one student.
# Higher values usually lead to passing, with a few realistic exceptions.
student_data = [
    [0, 35], [1, 45], [1, 70], [2, 50], [2, 85],
    [3, 55], [3, 90], [4, 60], [4, 80], [5, 65],
    [5, 92], [6, 70], [6, 85], [7, 65], [7, 95],
    [8, 75], [8, 90], [9, 80], [9, 98], [10, 90],
]
passed = [
    0, 0, 0, 0, 1,
    0, 1, 0, 1, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
]

# Step 2: Create and train a logistic-regression model using both features.
model = LogisticRegression()
model.fit(student_data, passed)

# Step 3: Save the trained model to a file so it can be used later.
model_filename = "student_pass_model.joblib"
joblib.dump(model, model_filename)

# Step 4: Load the saved model into a new variable.
# This variable contains the learned information from the earlier training.
loaded_model = joblib.load(model_filename)

# Step 5: Use the loaded model to predict outcomes for new student data.
test_students = [[1, 45], [5, 75], [9, 95]]
predictions = loaded_model.predict(test_students)

for student, prediction in zip(test_students, predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(
        f"{student[0]} study hour(s), "
        f"{student[1]}% attendance: {result}"
    )

# Step 6: Explain why loading the model is useful.
print("\nWe did not need to retrain the model because the saved file already stores what it learned from the training data.")
