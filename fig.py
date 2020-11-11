import config
import re
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from matplotlib.pyplot import MultipleLocator

fold=config.fold1
fold2=config.fold2

acc_pt,acc_ft=[[] for i in range(fold*fold2)],[[] for i in range(fold*fold2)]
kappa_pt,kappa_ft=[[] for i in range(fold*fold2)],[[] for i in range(fold*fold2)]
auprc_pt,auprc_ft=[[] for i in range(fold*fold2)],[[] for i in range(fold*fold2)]
auroc_pt,auroc_ft=[[] for i in range(fold*fold2)],[[] for i in range(fold*fold2)]



for i in range(fold):
    for j in range(fold2):
        with open('result_out{m}_{n}.out'.format(m=str(i+10),n=str(j+1)),'r') as f:
            for l in f.readlines():
                line=re.split(r'[,,(,),\s]',l)
                if line[0]=='pretrained:':
                    acc_pt[i*fold2+j].append(float(line[1]))
                    kappa_pt[i*fold2+j].append(float(line[3]))
                    auprc_pt[i*fold2+j].append(float(line[6]))
                    auroc_pt[i*fold2+j].append(float(line[8]))
                if line[0]=='no-pretrain:':
                    acc_ft[i*fold2+j].append(float(line[1]))
                    kappa_ft[i*fold2+j].append(float(line[3]))
                    auprc_ft[i*fold2+j].append(float(line[6]))
                    auroc_ft[i*fold2+j].append(float(line[8]))


plt.figure(figsize=[5,5],dpi=600)
x=np.linspace(0.6,1,100)
plt.plot(x,x,color='grey',linestyle='--')
line=np.zeros(l)+0.001
color=['g' for i in range(l)]
plt.scatter([np.mean([acc_ft[j][i] for j in range(fold)]) for i in range(l)],
            [np.mean([acc_pt[j][i] for j in range(fold)]) for i in range(l)],
            linewidths=line,c=color)
font={'size':30}
plt.ylabel('scPretrain',fontdict=font)
plt.xlabel('Without pre-training',fontdict=font)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Accuracy',fontsize=30)
plt.savefig('fig/nn_acc.png')

plt.figure(figsize=[5,5],dpi=600)
x=np.linspace(0,1,100)
plt.plot(x,x,color='grey',linestyle='--')
l=len(acc_pt[0])
line=np.zeros(l)+0.001
color=['g' for i in range(l)]
plt.scatter([np.mean([kappa_ft[j][i] for j in range(fold)]) for i in range(l)],
            [np.mean([kappa_pt[j][i] for j in range(fold)]) for i in range(l)],
            linewidths=line,c=color)
font={'size':30}
plt.ylabel('scPretrain',fontdict=font)
plt.xlabel('Without pre-training',fontdict=font)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Cohen\'s Kappa',fontsize=30)
plt.savefig('fig/nn_kappa.png')

l=len(acc_pt[0])
plt.figure(figsize=[5,5],dpi=600)
x=np.linspace(0.3,1,100)
plt.plot(x,x,color='grey',linestyle='--')
line=np.zeros(l)+0.001
color=['g' for i in range(l)]
plt.scatter([np.mean([auprc_ft[j][i] for j in range(fold)]) for i in range(l)],
            [np.mean([auprc_pt[j][i] for j in range(fold)]) for i in range(l)],
            linewidths=line,c=color)
font={'size':30}
plt.ylabel('scPretrain',fontdict=font)
plt.xlabel('Without pre-training',fontdict=font)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('AUPRC',fontsize=30)
plt.savefig('fig/nn_auprc.png')

plt.figure(figsize=[5,5],dpi=600)
x=np.linspace(0.4,1,100)
plt.plot(x,x,color='grey',linestyle='--')
line=np.zeros(l)+0.001
color=['g' for i in range(l)]
plt.scatter([np.mean([auroc_ft[j][i] for j in range(fold)]) for i in range(l)],
            [np.mean([auroc_pt[j][i] for j in range(fold)]) for i in range(l)],
           linewidths=line,c=color)
font={'size':30}
plt.ylabel('scPretrain',fontdict=font)
plt.xlabel('Without pre-training',fontdict=font)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('AUROC',fontsize=30)
plt.savefig('fig/nn_auroc.png')

