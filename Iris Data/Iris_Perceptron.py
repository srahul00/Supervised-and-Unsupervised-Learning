import numpy as np
import pandas as pd

df = pd.read_csv("D:/Spyder/Projects/Iris Data/iris.data")
df = df[:100]
print(df)

label = pd.Series(df['Label'])
label = label.map({'Iris-setosa' : 0, 'Iris-versicolor' : 1})

df = df.sample(frac = 1)
X = np.array(df.drop(labels = ['Label'], axis = 1))
y = np.array(label)

ntrn = 70
X_Train, X_Test = X[:ntrn], X[ntrn: ] 
Y_Train, Y_Test = y[:ntrn], y[ntrn: ]

weights = np.array([0.5, 0.5, 0.5, 0.5], dtype = np.float32)
yhat = np.ones(ntrn)

for epoch in range(1000):
    
    #for i in range(len(X_Train)):
        
    #if(Y_Train[i] == yhat[i]):
        #pass
    
    sums = np.dot(X_Train, weights)
    print(sums)
    for i, s in enumerate(sums):
        if(s >= 0):
            yhat[i] = 1
        else:
            yhat[i] = 0
    
    #if(Y_Train == yhat):
        #break
    else:
        weights += np.dot((Y_Train-yhat), X_Train)

ypred = []
for i in X_Test:
    
    s = np.dot(i, weights)
    if(s>=0):
        ypred.append(1)
    else:
        ypred.append(0)

count = 0
for i,j in zip(ypred, Y_Test):
    count += int(i==j)    
    
acc = count/len(ypred)
