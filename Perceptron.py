import numpy as np

# Sample Data: OR Gate
x = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 1, 1, 1])
yhat = np.array([1, 1, 1, 1])
weights = np.array([0.5, 0.5, 0.5])

while(sum(np.array(np.equal(y, yhat), dtype = np.int32)) != len(x)):    
   
    for i in range(len(x)):
        if(y[i] == yhat[i]):
            pass
        
        s = np.dot(x[i], weights)
        if(s >= 0):
            yhat[i] = 1
        else:
            yhat[i] = 0
        
        if(y[i] == yhat[i]):
            pass
        else:
            weights += (y[i]-yhat[i])*x[i]
            
print(f"Optimal Weights: {weights}")
