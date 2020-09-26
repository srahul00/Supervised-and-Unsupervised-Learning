import numpy as np
import pandas as pd

df = pd.read_csv("D:/Spyder/Projects/Iris Data/iris.data")
df = df[:100]
df = df.sample(frac = 1)
print(df.head())

label = pd.Series(df['Label'])
label = label.map({'Iris-setosa' : 0, 'Iris-versicolor' : 1})# , 'Iris-virginica' : 2})

X_train, y_train = df.iloc[:int(len(df)*.9), :-1].values, df.iloc[:int(len(df)*.9), -1].values
X_test, y_test   = df.iloc[int(len(df)*.9):, :-1].values, df.iloc[int(len(df)*.9):, -1].values
  

class Perceptron():

    def __init__(self):
        self.weights = np.array([.5, .5, .5, .5])
        self.bias = np.array([.5])

    def forward(self, x):
        pred = np.dot(x, self.weights) + self.bias
        pred = 1/(1 + np.exp(-pred))
        return pred
    
    def backward(self, x, yhat, y):
        err = .5*((y - yhat)**2)
        gradient_w = (yhat - y) * yhat * (1 - yhat) * x
        gradient_b = (yhat - y) * yhat * (1 - yhat)
        self.weights -= gradient_w
        self.bias -= gradient_b
        return err
    
    def fit(self, X_train, y_train, epochs = 50):
        for epoch in range(epochs):
            for x, y in zip(X_train, y_train):
                yhat = self.forward(x)
                err = self.backward(x, yhat, y)
                self.error += err
    
    def predict(self, X_test):
        pred = []
        for i in X_test:
            yhat = 1 if self.forward(i)>=0.5 else 0
            pred.append(yhat)
        return np.array(pred)
    
    def __str__(self):
        return str(self.weights)

clf = Perceptron()
clf.fit(X_train, y_train)
ypred = clf.predict(X_test)

TP, TN, FP, FN = 0, 0, 0, 0
for i, j in zip(y_test, ypred):
    
    if(i == 1 and i == j):
        TP += 1
    elif(i == j):
        TN += 1
        
    #ans 1 but pred not 1
    elif(i == 1):
        FN += 1
    else:
        FP += 1

acc = (TP+TN)/len(ypred)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
fmeasure = 2*(precision*recall)/(precision+recall)

print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {fmeasure}")
  