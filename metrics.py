from sklearn.metrics import accuracy_score,silhouette_score
from sklearn.metrics import precision_recall_curve,roc_auc_score,auc
import numpy as np

def kappa(out,target,label_num):
    matrix=np.zeros((label_num,label_num))
    for o,t in zip(out,target):
        matrix[o][t]+=1
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)

def multi_auprc_auroc(out,target,label_num):
    auprc=[]
    auroc=[]
    for i in range(label_num):
        out_new=[]
        target_new=[]
        for o,t in zip(out[:,i],target):
            out_new.append(o)
            target_new.append(int(t==i))
        auroc.append(float(roc_auc_score(target_new,out_new)))
        pre,re,_=precision_recall_curve(target_new,out_new)
        auprc.append(float(auc(re,pre)))
    return np.mean(auprc),np.mean(auroc)