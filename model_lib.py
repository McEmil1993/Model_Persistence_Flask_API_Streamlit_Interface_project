"""A tiny logistic regression using only numpy (no scikit-learn needed).

Kept in its own module so joblib can save and load it from any script.
"""
import numpy as np


class LogisticModel:
    def __init__(self, lr=0.1, epochs=5000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Standardise features so gradient descent behaves well
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Z = (X - self.mean_) / self.std_
        self.coef_ = np.zeros(Z.shape[1])
        self.intercept_ = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(Z @ self.coef_ + self.intercept_)
            error = p - y
            self.coef_ -= self.lr * (Z.T @ error) / len(y)
            self.intercept_ -= self.lr * error.mean()
        return self

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        """Return the probability of PASS for each row."""
        X = np.asarray(X, dtype=float)
        Z = (X - self.mean_) / self.std_
        return self._sigmoid(Z @ self.coef_ + self.intercept_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
