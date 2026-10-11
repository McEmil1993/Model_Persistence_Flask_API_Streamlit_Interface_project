# Why the model does not need to be trained again

Training (`model.fit(X, y)`) is the step where the model *learns* its
parameters from the data. For logistic regression those parameters are the
**coefficient** (how much each extra study hour changes the odds of passing)
and the **intercept**.

`joblib.dump()` writes the trained model object - including these learned
parameters - to `student_pass_model.joblib`. `joblib.load()` rebuilds the
exact same object in memory, with the parameters already filled in.

Making a prediction only plugs new numbers into the stored formula:

    probability = 1 / (1 + e^-(intercept + coefficient * study_hours))

No learning is needed for that, so `predict()` works immediately after
loading. Saving the model also means:

* predictions are fast (no waiting for training),
* results are consistent (same model every time),
* an API or app can use the model without access to the training data.

You only retrain when you have new data or want to change the features.