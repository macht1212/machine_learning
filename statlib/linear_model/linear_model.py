import numpy as np


class LinearRegression:

    def __init__(self):
        self.w = None
        self.lr = .1
        self.eps = .0001
        self.max_iter = 100
        self.X = None

    def _X_train_prep(self, X_):
        self.X = np.column_stack([X_, np.ones(X_.shape[0])])

    def _x_test_prep_(self, X):
        return np.column_stack([X, np.ones(X.shape[0])])

    def _mse(self, y):
        y_ = self.X @ self.w
        return np.sum((y - y_) ** 2) / len(y_)

    def _gr_mse(self, y):
        y_ = self.X @ self.w
        return 2 / len(self.X) * (y - y_) @ (-self.X)

    def _init_w(self):
        return np.zeros(self.X.shape[1])

    def fit(self, X, y):
        X = X.copy()
        self._X_train_prep(X)

        self.w = self._init_w()

        next_w = self.w

        for _ in range(self.max_iter):
            cur_w = next_w

            next_w = cur_w - self.lr * self._gr_mse(y)
            self.w = cur_w

            if np.linalg.norm(cur_w - next_w, ord=2) <= self.eps:
                break

    def get_coefs(self):
        return self.w[:-1]

    def get_w0(self):
        return self.w[-1]

    def predict(self, X):
        x = self._x_test_prep_(X)
        return x @ self.w


class LogisticRegression(LinearRegression):

    def __init__(self):
        super().__init__()

    def _gr_log_loss(self, y):
        y_prob = self._sigmoid(self.X @ self.w)
        return self.X.T @ (y_prob - y)

    def _sgr_log_loss(self, y):

    def fit(self, X, y):
        X = X.copy()
        self._X_train_prep(X)

        self.w = self._init_w()

        next_w = self.w

        for _ in range(self.max_iter):
            cur_w = next_w

            next_w = cur_w - self.lr * self._gr_log_loss(y)
            self.w = cur_w

            if np.linalg.norm(cur_w - next_w, ord=2) <= self.eps:
                break

    @staticmethod
    def _sigmoid(L):
        return 1 / (1 + np.exp(-L))

    def predict_proba(self, X):
        X = self._x_test_prep_(X)
        return self._sigmoid(X @ self.w)

    def predict(self, X):
        x = self.predict_proba(X)
        return np.where(x > .5, 1, 0)

