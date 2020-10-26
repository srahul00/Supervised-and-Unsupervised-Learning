import numpy as np
import pandas as pd


class GaussianClassifier():

    def __init__(self):
        pass

    def fit(self, X_train, y_train):

        self.mean = []
        self.cov = []
        self.prior = []
        self.cov = np.cov(X_train[indices].T)
        self.numofclasses = len(np.unique(y_train))
        for i in range(self.numofclasses):
            indices = y_train == i
            self.mean.append(np.mean(X_train[indices], axis=0))
            self.prior.append(1.0*indices.sum()/len(X_train))

    def predict(self, X_test):

        probabilities = np.array([])
        for i in range(self.numofclasses):
            print((X_test-self.mean[i]).shape, np.linalg.inv(self.cov).shape, (X_test-self.mean[i]).T.shape)
            likelihood = np.exp(-0.5*(X_test-self.mean[i])@np.linalg.inv(self.cov)@(X_test-self.mean[i]).T)/np.sqrt((2*np.pi)**X_test.shape[0]*np.linalg.det(self.cov[i]))
            print(likelihood)
            probabilities = np.append(probabilities, likelihood*self.prior[i])
            # print(likelihood)

        print(probabilities.shape)
        probabilities = probabilities/np.sum(probabilities, axis=1)
        ypred = np.argmax(probabilities, axis=1)
        return ypred

def main():

    df = pd.read_csv("iris.data")
    df = df.sample(frac = 1)
    # print(df.head())

    label = df.pop('Label')
    label = label.map({'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2})

    test_size = 0.20
    train_till_row = int(len(df)*(1 - test_size))
    X_train, y_train = pd.DataFrame(df[ :train_till_row].values), label[ :train_till_row].values
    X_test, y_test   = pd.DataFrame(df[train_till_row: ].values), label[train_till_row: ].values

    #print(np.cov(X_train.T))
    clf = GaussianClassifier()
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)

    print(f"Accuracy: {(y_test == ypred).sum()/len(ypred)*100}")


if __name__ == '__main__':
    main()