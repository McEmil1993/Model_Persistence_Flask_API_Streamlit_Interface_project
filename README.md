# Student Pass Predictor - LITE (numpy only)

No scikit-learn, scipy or pandas needed for Exercises 1 and 2.

## Termux setup
```
pkg install python python-pip python-numpy
pip install joblib flask
```

## Exercise 1 - Model persistence
```
python train_model.py
python load_and_predict.py
```
Why no retraining? Training (fit) learns the model's coefficients and
intercept. joblib.dump() saves those learned numbers to disk. joblib.load()
restores the same fitted model, so predict() only applies the stored numbers
to new input. Training is needed again only if the data or features change.

## Exercise 2 - Flask API
Session 1:  python app.py
Session 2:  bash test_api.sh
The attendance model is trained on study hours + attendance (train_model.py),
so attendance really affects the prediction. Attendance outside 0-100
returns HTTP 400.

## Exercise 3 - Streamlit (optional on Termux, needs pandas + streamlit)
streamlit run streamlit_app.py
If Streamlit will not install on your phone, run it on a PC or Google Colab.
