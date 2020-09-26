import numpy as np
import pandas as pd


class KNNC():

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
            labels = np.zeros(len(np.unique(self.y)))
            for cls in self.y[indices]:
                labels[cls] += 1
            pred.append(np.argmax(labels))
        return np.array(pred)


def main():

    df = pd.read_csv("iris.data")
    df = df.sample(frac = 1)
    # print(df.head())

    label = df.pop('Label')
    label = label.map({'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2})

    test_size = 0.20
    train_till_row = int(len(df)*(1 - test_size))
    X_train, y_train = df[ :train_till_row].values, label[ :train_till_row].values
    X_test, y_test   = df[train_till_row: ].values, label[train_till_row: ].values

    clf = KNNC()
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)

    print(f"Accuracy: {(y_test == ypred).sum()/len(ypred)*100}")


if __name__ == '__main__':
    main()