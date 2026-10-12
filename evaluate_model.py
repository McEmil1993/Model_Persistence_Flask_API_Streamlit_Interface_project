import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Same training data as train_model.py
# Columns: [study_hours, attendance_percentage]
X = np.array([
    [1, 50], [2, 55], [3, 60], [3, 65], [4, 70],
    [5, 75], [6, 80], [7, 85], [8, 90], [9, 95]
])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 0 = fail, 1 = pass


def evaluate():
    """Return the confusion matrix and metrics using 3-fold cross-validation.

    Each student is predicted by a model that did NOT see that student
    during training, so the score is more honest than testing on the
    training data itself.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_pred = cross_val_predict(LogisticRegression(), X, y, cv=cv)

    # scikit-learn order: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }


if __name__ == "__main__":
    r = evaluate()
    print("Confusion Matrix (rows = actual, columns = predicted)")
    print("               Pred Pass  Pred Fail")
    print(f"Actual Pass    TP = {r['TP']:<4}   FN = {r['FN']}")
    print(f"Actual Fail    FP = {r['FP']:<4}   TN = {r['TN']}")
    print()
    print(f"Accuracy : {r['accuracy']:.2f}")
    print(f"Precision: {r['precision']:.2f}")
    print(f"Recall   : {r['recall']:.2f}")
    print(f"F1-Score : {r['f1']:.2f}")