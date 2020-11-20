import numpy as np
import pandas as pd


class KMeans():

    def __init__(self, K=8):
        self.nclusters = K

    def fit(self, X, maxiter=10):

        indices = np.random.choice(X.shape[0], self.nclusters, replace=False)
        self.centroids = X[indices]
        X = X[list(set(np.arange(X.shape[0])) - set(indices))]

        while(1):
            classes = np.empty_like((X.shape[0], self.nclusters), dtype=np.float32)
            counter = [0]*self.nclusters
            for data in X:
                dist = np.linalg.norm(data-self.centroids)
                classindex = np.argmin(dist)
                classes[counter[classindex], classindex] = data
                counter[classindex] += 1

            centroids = np.array([])
            for i in range(self.nclusters):
                print(i, classes[i])
                centroids = np.append(centroids, np.mean(classes[i]))
            print(centroids)
            convergence = 1
            for i in range(len(centroids)):
                if((centroids[i] - self.centroids[i]).sum() != 0):
                    self.centroids = centroids
                    convergence = 0
            if convergence:
                break

    def predict(self, data):
        pass


def main():

    df = pd.read_csv("iris.data")
    df = df.iloc[:100, :]
    print(df.head())
    df = df.sample(frac = 1)

    label = df.pop('Label')
    label = label.map({'Iris-setosa' : 0, 'Iris-versicolor' : 1})#, 'Iris-virginica' : 2})

    test_size = 0.20
    train_till_row = int(len(df)*(1 - test_size))
    X_train, y_train = df[ :train_till_row].values, label[ :train_till_row].values
    X_test, y_test   = df[train_till_row: ].values, label[train_till_row: ].values

    clf = KMeans(K=2)
    clf.fit(X_train)
    ypred = clf.predict(X_test)

    print(f"Accuracy: {(y_test == ypred).sum()/len(ypred)*100}")


if __name__ == '__main__':
    main()