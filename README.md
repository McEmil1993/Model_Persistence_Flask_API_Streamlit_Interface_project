# Student Prediction

## Run the Streamlit app

Install the dependencies, then start Streamlit from this directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Do not run the app with `python streamlit_app.py`; Streamlit needs to run it
through its CLI for session state and browser rendering to work correctly.

The saved models were trained with scikit-learn 1.9.1. If you change that
version, retrain the models with `python train_model.py` before loading them.
