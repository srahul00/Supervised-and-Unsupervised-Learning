import numpy as np
from sklearn.linear_model import LinearRegression

import sys
sys.path.append("./Models/")
from KNN import KNNR


def rmse(ypred, y):

    error = ypred - y
    # mse = rsse^2 / n
    mse = (np.linalg.norm(error)**2)/len(error)
    rmse = mse**0.5
    return rmse


def main():

    # Toy Example: Hours Studied vs SAT Score
    X = np.array([4, 9, 10, 14, 4, 7, 12, 22, 1, 3, 8, 11, 5, 6, 10, 11, 16, 13, 13, 10], dtype=np.float64)
    y = np.array([390, 580, 650, 730, 410, 530, 600, 790, 350, 400, 590, 640, 450, 520, 690, 690, 770, 700, 730, 640], dtype=np.float64)

    # Shuffling
    order = np.random.permutation(len(X))
    X = X[order].reshape((-1, 1))
    y = y[order]

    test_size = 0.20
    train_till_row = int(len(X)*(1 - test_size))
    X_train, y_train = X[ :train_till_row, :], y[ :train_till_row]
    X_test, y_test = X[train_till_row: , :], y[train_till_row: ]

    clf1 = KNNR()
    clf1.fit(X_train, y_train)
    yKNN = clf1.predict(X_test)

    clf2 = LinearRegression()
    clf2.fit(X_train, y_train)
    yLinear = clf2.predict(X_test)

    print(f"RMSE for KNN: {rmse(yKNN, y_test)}")
    print(f"RMSE for Linear Regression: {rmse(yLinear, y_test)}")
    
    '''
    clf3 = KNNR()
    clf3.fit(X, y)
    ans1 = clf3.predict([[15]])
    ans2 = clf3.predict([[14.5]])
    print(f"15 Hours: {ans1} \n14.5 Hours: {ans2}")
    '''

if __name__ == '__main__':
    main()