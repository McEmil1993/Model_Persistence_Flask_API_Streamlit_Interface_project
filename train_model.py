import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

def train_and_save_model():
    print("--- EXERCISE 1: MODEL PERSISTENCE ---")
    
    # 1. Load data
    print("Loading dataset...")
    data = pd.read_csv("student_data.csv")
    
    # 2. Separate features and target
    X = data[["study_hours"]]
    y = data["result"]
    
    # 3. Train the model
    print("Training the Decision Tree model...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    print("[OK] Model trained successfully.")
    
    # 4. Save the model
    model_filename = "student_pass_model.joblib"
    joblib.dump(model, model_filename)
    print(f"[OK] Model saved as '{model_filename}'")
    
    # 5. Load the model back
    print("\nLoading the saved model to verify...")
    loaded_model = joblib.load(model_filename)
    print("[OK] Model loaded successfully.")
    
    # 6. Make 3 predictions
    test_hours = [2, 5, 8]
    print("\n--- 3 Sample Predictions ---")
    for hours in test_hours:
        sample = pd.DataFrame({"study_hours": [hours]})
        prediction = loaded_model.predict(sample)[0]
        print(f"Study Hours: {hours} -> Predicted Result: {prediction}")
        
    # 7. Explanation
    print("\n--- EXPLANATION ---")
    print(
        "The model does not need to be trained again because the .joblib file "
        "stores the learned parameters (the tree structure). When we load it, "
        "we are simply restoring those parameters to make predictions. "
        "Retraining is only needed if we have new data or change the algorithm."
    )

if __name__ == "__main__":
    train_and_save_model()