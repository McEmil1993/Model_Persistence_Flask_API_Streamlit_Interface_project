# Student Performance Prediction Exercises

This folder contains one project completing all three exercises. The sample CSV is synthetic and illustrative; it is not suitable for real student decisions.

## Project structure

```text
activity/
├── .streamlit/config.toml     # Theme and Streamlit app chrome
├── data/                      # Training dataset
├── models/                    # Persisted model used by both interfaces
├── postman/                   # Importable API test collection
├── student_performance/       # Model, training, API, and UI modules
├── static/fonts/               # Self-hosted Geist fonts and OFL license
├── tests/                     # Automated regression checks
├── requirements.txt           # Direct project dependencies
└── streamlit_app.py           # Streamlit entry point
```

`activity-preview.png` is an optional screenshot. `.venv/` and `__pycache__/` are generated local files and are ignored by Git; remove them only when you are comfortable recreating the environment or cache.

### What is essential?

- Keep `data/`, `student_performance/`, `streamlit_app.py`, `requirements.txt`, and `.streamlit/config.toml` to retrain and run the project.
- Keep `models/student_pass_model.joblib` to run predictions without retraining. It can be regenerated from the CSV.
- `tests/`, `postman/`, and this README are not needed to launch the app, but are useful evidence of quality, API behavior, and how to run/present the work.
- `requirements.txt` pins the direct dependencies used by the project. Removing `requirements.lock` is safe; transitive dependencies may resolve to newer versions on future installs.
- `static/fonts/` contains the self-hosted Geist fonts and SIL Open Font License, so the typography does not depend on an external font service.

## Setup

Python 3.12 is supported. Create and activate a virtual environment, then install the pinned direct dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate`.

## Train and persist the model

```bash
python -m student_performance.train_model
```

This reads `data/student_performance.csv`, fits the preprocessing and classifier pipeline, and saves `models/student_pass_model.joblib`. Run training again only when the training data or model design changes. The prediction scripts load and reuse the saved artifact.

## Run the Flask API

```bash
flask --app student_performance.app:create_app run
```

Send a `POST` request to `http://127.0.0.1:5000/predict` with JSON:

```json
{
  "study_hours": 5,
  "attendance_percentage": 90
}
```

From PowerShell, the same request can be sent with:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data-raw '{"study_hours":5,"attendance_percentage":90}'
```

The API validates attendance from 0 through 100 and returns both inputs and the prediction. The current model uses study hours only, so attendance does not affect the predicted class.

## Run the Streamlit interface

```bash
streamlit run streamlit_app.py
```

The student name is displayed only; the model predicts from study hours. The chart shows pass probability across the full study-hour range and a 50% decision threshold.

The interface uses Geist for UI text, Geist Pixel Square for headings, and Geist Mono for code. These self-hosted open fonts match the Bryllim reference without external font requests.

## Test the API with Postman

1. Start the Flask API using the command above and leave that terminal running.
2. In Postman, import `postman/student-performance-api.postman_collection.json`.
3. Open **Student Performance API → Predict with valid inputs** and send the included JSON body. The expected response is HTTP 200 with both input values and a `Pass` or `Fail` prediction.
4. Run the collection to also verify the invalid-attendance and missing-field cases return HTTP 400 with JSON errors.

The collection's `baseUrl` defaults to `http://127.0.0.1:5000`; change it if you start Flask on another port. For a terminal-runner alternative, use Newman:

```bash
npx --yes newman run postman/student-performance-api.postman_collection.json
```

## Presentation flow

1. **Data:** show the CSV and explain that it is a small synthetic example.
2. **Training:** `python -m student_performance.train_model` loads the data, fits preprocessing and logistic regression, then saves one reusable model artifact.
3. **Prediction:** the demo script and Flask endpoint load that same artifact; they do not retrain for each request.
4. **API:** Flask validates the JSON contract, calls the model with `study_hours`, and returns a JSON class. Attendance is validated and returned to demonstrate the requested input, but is not a model feature in this exercise.
5. **UI:** Streamlit lets a person enter a display-only name and study hours, then shows the prediction and a chart of predicted outcomes by study time.
6. **Verification:** show the three Postman requests and the automated Python tests.

This separation keeps model training, inference, API validation, and presentation concerns independent. A saved scikit-learn pipeline keeps preprocessing bundled with the classifier, while Flask and Streamlit provide API and human-facing entry points to the same prediction logic.

## Run checks

```bash
python -m unittest discover -s tests -v
```

Only load the generated Joblib model from a trusted local source; Joblib artifacts use Python serialization and are not safe to load from untrusted sources.
