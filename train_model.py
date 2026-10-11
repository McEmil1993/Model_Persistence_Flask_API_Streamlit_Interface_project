
"""
Exercise 1: Model Persistence and Model Comparison

Creates a simulated student dataset, compares three classifiers,
selects the best model using cross-validation, evaluates it on a
separate test set, and saves the trained model using joblib.

IMPORTANT:
The dataset and labels are simulated for educational purposes.
They are not real student performance records.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_pass_model.joblib"
DATASET_PATH = BASE_DIR / "student_demo_dataset.csv"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"


# --------------------------------------------------
# 2. Create a simulated demonstration dataset
# --------------------------------------------------

def create_demo_dataset(number_of_students=400, random_state=42):
    rng = np.random.default_rng(random_state)

    # Study hours: 0 to 24 hours per day
    study_hours = rng.uniform(0, 24, number_of_students)

    # Attendance: 0 to 100 percent
    attendance = rng.uniform(0, 100, number_of_students)

    # Illustrative scoring rule:
    # Both study hours and attendance contribute to the score.
    simulated_score = (
        0.55 * (study_hours / 24)
        + 0.45 * (attendance / 100)
        + rng.normal(0, 0.07, number_of_students)
    )

    # These labels are simulated, not actual student outcomes.
    results = np.where(simulated_score >= 0.52, "Pass", "Fail")

    dataset = pd.DataFrame({
        "StudyHours": np.round(study_hours, 2),
        "Attendance": np.round(attendance, 2),
        "Result": results,
    })

    dataset.to_csv(DATASET_PATH, index=False)

    return dataset


# --------------------------------------------------
# 3. Train and compare candidate models
# --------------------------------------------------

def main():
    dataset = create_demo_dataset()

    print("=" * 55)
    print("STUDENT PERFORMANCE DEMONSTRATION DATASET")
    print("=" * 55)
    print(dataset.head(10).to_string(index=False))
    print(f"\nTotal records: {len(dataset)}")
    print("\nClass distribution:")
    print(dataset["Result"].value_counts())
    print(f"\nDataset saved to: {DATASET_PATH.name}")

    # Use both features, in this exact order.
    X = dataset[["StudyHours", "Attendance"]]
    y = dataset["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Scaling is included in the pipelines so it is fitted
    # separately inside each cross-validation training fold.
    models = {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(
                max_depth=5,
                random_state=42,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
            )),
        ]),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    comparison_results = []
    trained_models = {}

    print("\n" + "=" * 55)
    print("CROSS-VALIDATION MODEL COMPARISON")
    print("=" * 55)

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
        )

        comparison_results.append({
            "Model": name,
            "Mean CV F1": scores.mean(),
            "CV F1 Standard Deviation": scores.std(),
        })

        trained_models[name] = model

        print(
            f"{name}: mean F1 = {scores.mean():.4f}, "
            f"standard deviation = {scores.std():.4f}"
        )

    comparison = pd.DataFrame(comparison_results)
    comparison = comparison.sort_values(
        by="Mean CV F1",
        ascending=False,
    )

    comparison.to_csv(COMPARISON_PATH, index=False)

    # Select the model with the highest mean cross-validation F1.
    best_name = comparison.iloc[0]["Model"]
    best_model = trained_models[best_name]

    # Fit the selected pipeline on the complete training split.
    best_model.fit(X_train, y_train)

    # Evaluate once on the held-out test split.
    test_predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(
        y_test, test_predictions, pos_label="Pass", zero_division=0
    )
    recall = recall_score(
        y_test, test_predictions, pos_label="Pass", zero_division=0
    )
    f1 = f1_score(
        y_test, test_predictions, pos_label="Pass", zero_division=0
    )

    print("\n" + "=" * 55)
    print("SELECTED MODEL AND TEST RESULTS")
    print("=" * 55)
    print(f"Selected model: {best_name}")
    print(f"Test accuracy:  {accuracy:.4f}")
    print(f"Pass precision: {precision:.4f}")
    print(f"Pass recall:    {recall:.4f}")
    print(f"Pass F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    ))

    # Save the fitted pipeline, including its scaler.
    # The API and Streamlit app must supply both features.
    joblib.dump(best_model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH.name}")
    print(f"Comparison saved to: {COMPARISON_PATH.name}")

    # Example predictions for two hypothetical students.
    examples = pd.DataFrame({
        "StudyHours": [5, 5],
        "Attendance": [100, 85],
    })

    example_predictions = best_model.predict(examples)

    print("\nExample predictions from the selected model:")
    for (_, row), prediction in zip(
        examples.iterrows(),
        example_predictions,
    ):
        print(
            f"Study hours: {row['StudyHours']:g}, "
            f"Attendance: {row['Attendance']:g}% "
            f"-> Predicted result: {prediction}"
        )

    print(
        "\nReminder: these results evaluate a simulated dataset. "
        "They do not establish real-world student prediction accuracy."
    )


if __name__ == "__main__":
    main()