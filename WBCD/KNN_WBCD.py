import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../Models/")
from KNN import KNNC


def aucplot(model_probabilities, ytest):

    assert len(model_probabilities) == len(ytest)
    # sorting descending order of model prob
    indices = np.argsort(model_probabilities)[::-1]
    model_probabilities = model_probabilities[indices]
    ytest = ytest[indices]

    # len(fpr or tpr) = 0 + len(ytest) + 1
    tpr = np.zeros(len(ytest)+1)
    fpr = np.zeros(len(ytest)+1)
    tp, fp = 0, 0
    for i in ytest:
        if(i == 1):
            tp += 1
        else:
            fp += 1
        fpr[fp+tp] = fp
        tpr[fp+tp] = tp

    tpr = tpr/tpr[-1]
    fpr = fpr/fpr[-1]
    tpr = np.append(tpr, 1)
    fpr = np.append(fpr, 1)
    tpr = [tp for fp, tp in sorted(zip(fpr, tpr))]
    fpr.sort()
    auc = 0
    for i in range(len(fpr) - 1):
        auc += 0.5*(fpr[i+1]-fpr[i])*(tpr[i] + tpr[i+1])

    print(f"AUC: {auc}")
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Area Under ROC Curve")
    plt.legend([f"AUC: {auc}"])
    plt.show()


def BareNuclei(x):
    if x == '?':
        return 0
    else:
        return int(x)


def main():

    data = pd.read_csv('breast-cancer-wisconsin.data', header=None)
    data.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 
                    'Mitoses', 'Class']

    # print(data.head())
    data.drop(['Sample code number'], axis=1, inplace=True)
    # Changing string '?' to integer 0
    data['Bare Nuclei'] = data['Bare Nuclei'].apply(BareNuclei)
    # Changing Class values from 4->1 and 2->0
    data['Class'] = data['Class'].map({4: 1, 2: 0})
    # data = data.sample(frac = 1)
    # print(data.info())

    test_size = 0.25
    train_till_row = int(len(data)*(1 - test_size))
    X_train = data[: train_till_row].drop(['Class'], axis=1).values
    y_train = data[: train_till_row]['Class'].values
    X_test = data[train_till_row: ].drop(['Class'], axis=1).values
    y_test = data[train_till_row: ]['Class'].values

    for kval in range(1, 3, 2):
        clf = KNNC(kval)
        clf.fit(X_train, y_train)
        ypred = clf.predict(X_test)

        # 0 - Benign
        # 1 - Malignant
        TP, TN, FP, FN = 0, 0, 0, 0
        for i, j in zip(ypred, y_test):
            # print(i, j)
            if(i == 1 and j == 1):
                TP += 1
            elif(i == j):
                TN += 1

            # ans 1 but pred not 1
            elif(i == 0):
                FN += 1
            else:
                FP += 1

        acc = (TP+TN)/len(ypred)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        fmeasure = 2*(precision*recall)/(precision+recall)

        print(f"Hyperparameter K: {kval}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F-Measure: {fmeasure}\n")

        model_probabilities = clf.return_proba()
        aucplot(model_probabilities, y_test)


if __name__ == '__main__':
    main()
