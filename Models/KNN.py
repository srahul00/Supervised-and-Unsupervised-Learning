import numpy as np

# KNN Classification Class
class KNNC():

    def __init__(self, K=5):
        self.K = K
        self.proba = []

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            self.X = X
            self.y = y
        else:
            self.X = np.array(X)
            self.y = np.array(y)

    def predict(self, data):
        pred = []
        for row in data:
            dist = self.X - row
            dist = dist**2
            sums = np.sum(dist, axis=1)
            sums = np.sqrt(sums)

            indices = np.argsort(sums)[:self.K]
            positives = self.y[indices].sum()
            negatives = self.K - positives

            self.proba.append(positives/self.K)
            if(positives > negatives):
                pred.append(1)
            else:
                pred.append(0)

        return np.array(pred)

    def return_proba(self):
        return np.array(self.proba)


# KNN Regression Class
class KNNR():

    def __init__(self, K=5):
        self.K = K

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            self.X = X
            self.y = y
        else:
            self.X = np.array(X)
            self.y = np.array(y)

    def predict(self, data):
        pred = []
        for row in data:
            dist = self.X - row
            dist = dist**2
            sums = np.sum(dist, axis=1)
            sums = np.sqrt(sums)

            indices = np.argsort(sums)[:self.K]
            pred.append(self.y[indices].mean())

        return np.array(pred)
