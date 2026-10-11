import unittest

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_app_renders_prediction_controls_without_script_errors(self):
        app = AppTest.from_file("../streamlit_app.py").run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertIn("STUDENT PERFORMANCE", [item.value for item in app.title])
        self.assertEqual(len(app.text_input), 1)
        self.assertEqual(len(app.number_input), 1)
        self.assertIn("Generate estimate", [button.label for button in app.button])

    def test_prediction_displays_result_explanation_and_chart(self):
        app = AppTest.from_file("../streamlit_app.py").run(timeout=10)
        app.text_input[0].set_value("Ana")
        app.number_input[0].set_value(8)
        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Ana — Pass", [item.value for item in app.subheader])
        self.assertTrue(any("study hours only" in item.value for item in app.markdown))
        chart = app.get("vega_lite_chart")[0]
        self.assertIn('"type": "line"', chart.proto.spec)
        self.assertIn("PASS PROBABILITY BY STUDY TIME", [item.value for item in app.subheader])


if __name__ == "__main__":
    unittest.main()
