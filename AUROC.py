import matplotlib.pyplot as plt

fpr = [0]
tpr = [0]

modelProb = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37, 0.36, 
             0.35, 0.34, 0.33, 0.3, 0.1]
y = [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]

totalPos = sum(y)
totalNeg = len(y) - sum(y)

tp, fp = 0, 0
for i in range(len(modelProb)):
    if(y[i] == 1):
        tp += 1
    else:
        fp += 1
    tpr.append(tp/totalPos)
    fpr.append(fp/totalNeg)

tpr.append(1)
fpr.append(1)
tpr = [tp for fp, tp in sorted(zip(fpr, tpr), key = lambda x: x[0])]
fpr.sort()

auc = 0
for i in range(len(fpr) - 1):
    auc += 0.5 * (fpr[i+1] - fpr[i]) * (tpr[i] + tpr[i+1])
    
print(f"AUC : {auc}")

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend([f"AUC : {auc}"])
plt.show()

