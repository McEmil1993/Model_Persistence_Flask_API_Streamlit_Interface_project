import tempfile
import unittest
from pathlib import Path

from student_performance.app import create_app
from student_performance.training import train_model


TRAINING_DATA = """study_hours,outcome
0,Fail
1,Fail
2,Fail
4,Pass
6,Pass
8,Pass
"""


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        data_path = root / "students.csv"
        model_path = root / "student_pass_model.joblib"
        data_path.write_text(TRAINING_DATA, encoding="utf-8")
        train_model(data_path, model_path)
        self.client = create_app(model_path=model_path).test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_returns_both_inputs_and_model_prediction(self):
        response = self.client.post(
            "/predict",
            json={"study_hours": 5, "attendance_percentage": 90},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.get_json()),
            {"study_hours", "attendance_percentage", "prediction"},
        )
        self.assertEqual(response.get_json()["attendance_percentage"], 90)

    def test_rejects_attendance_outside_zero_to_one_hundred(self):
        response = self.client.post(
            "/predict",
            json={"study_hours": 5, "attendance_percentage": 101},
        )

        self.assertEqual(response.status_code, 400)

    def test_rejects_missing_or_extra_fields(self):
        for payload in (
            {"study_hours": 5},
            {"study_hours": 5, "attendance_percentage": 90, "name": "Student"},
        ):
            with self.subTest(payload=payload):
                response = self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 400)

    def test_returns_json_when_request_body_exceeds_limit(self):
        response = self.client.post(
            "/predict",
            data=b" " * 5000,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json(), {"error": "Request body is too large."})


if __name__ == "__main__":
    unittest.main()
