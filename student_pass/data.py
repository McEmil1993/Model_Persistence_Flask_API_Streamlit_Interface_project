"""Small sample dataset shared by the training scripts."""
import pandas as pd

def load_data():
    rows = [
        # study_hours, attendance_percentage, passed
        (0.5, 40, 0), (1.0, 55, 0), (1.5, 50, 0), (2.0, 60, 0),
        (2.5, 45, 0), (3.0, 70, 0), (3.5, 55, 0), (4.0, 65, 1),
        (4.5, 80, 1), (5.0, 75, 1), (5.5, 85, 1), (6.0, 90, 1),
        (6.5, 80, 1), (7.0, 95, 1), (8.0, 88, 1), (9.0, 92, 1),
        (1.0, 90, 0), (2.0, 95, 0), (5.0, 50, 1), (3.0, 100, 1),
    ]
    return pd.DataFrame(rows, columns=["study_hours", "attendance_percentage", "passed"])
