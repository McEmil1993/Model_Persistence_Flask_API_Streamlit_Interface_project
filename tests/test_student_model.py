from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from student_performance.model import load_model, predict_outcome
from student_performance.train_model import main
from student_performance.training import train_model


TRAINING_DATA = """study_hours,outcome
0,Fail
1,Fail
2,Fail
4,Pass
6,Pass
8,Pass
"""


class StudentModelTests(unittest.TestCase):
    def test_training_script_reloads_artifact_and_prints_three_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "students.csv"
            model_path = root / "student_pass_model.joblib"
            data_path.write_text(TRAINING_DATA, encoding="utf-8")
            output = StringIO()

            with (
                patch("student_performance.train_model.DEFAULT_DATA_PATH", data_path),
                patch("student_performance.train_model.DEFAULT_MODEL_PATH", model_path),
                redirect_stdout(output),
            ):
                main()

            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertIn("Saved trained model", lines[0])
            self.assertIn("2 study hours", lines[1])
            self.assertIn("5 study hours", lines[2])
            self.assertIn("8 study hours", lines[3])

    def test_training_saves_a_model_that_can_predict_after_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "students.csv"
            model_path = root / "student_pass_model.joblib"
            data_path.write_text(TRAINING_DATA, encoding="utf-8")

            train_model(data_path, model_path)
            saved_model = load_model(model_path)
            results = [predict_outcome(saved_model, hours) for hours in (1, 5, 8)]

            self.assertTrue(model_path.is_file())
            self.assertEqual(len(results), 3)
            self.assertTrue(set(results) <= {"Pass", "Fail"})

    def test_prediction_rejects_negative_and_non_finite_study_hours(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "students.csv"
            model_path = root / "student_pass_model.joblib"
            data_path.write_text(TRAINING_DATA, encoding="utf-8")
            train_model(data_path, model_path)
            model = load_model(model_path)

            for hours in (-1, float("nan"), float("inf")):
                with self.subTest(hours=hours):
                    with self.assertRaises(ValueError):
                        predict_outcome(model, hours)


if __name__ == "__main__":
    unittest.main()
