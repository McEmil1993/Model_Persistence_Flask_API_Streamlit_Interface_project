"""A tiny logistic regression written with numpy only (no scikit-learn needed).
It has the same methods as scikit-learn: fit, predict, predict_proba."""
import numpy as np


class LogisticModel:
    def __init__(self, lr=0.1, epochs=5000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Standardise features so gradient descent is stable
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_

        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(Xs @ self.coef_ + self.intercept_)
            error = p - y
            self.coef_ -= self.lr * (Xs.T @ error) / len(y)
            self.intercept_ -= self.lr * error.mean()
        return self

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        p = self._sigmoid(Xs @ self.coef_ + self.intercept_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
