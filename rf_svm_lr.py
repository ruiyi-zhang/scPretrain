import h5py
import pickle
import torch
import torch.nn as nn
import config
import random
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from data import dataset_name,get_gene_list,unlabelled
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve,roc_auc_score,auc

def multi_auprc_auroc(out,target,label_num):
    auprc=[]
    auroc=[]
    for i in range(label_num):
        out_new=[]
        target_new=[]
        for o,t in zip(out,target):
            out_new.append(int(o==i))
            target_new.append(int(t==i))
        auroc.append(float(roc_auc_score(target_new,out_new)))
        pre,re,_=precision_recall_curve(target_new,out_new)
        auprc.append(float(auc(re,pre)))
    return np.mean(auprc),np.mean(auroc)

pretrain_name=(dataset_name['mouse']+dataset_name['human'])

with open('pretrain_3clus_out18.pt','rb') as f:
    embed=torch.load(f)

class encoder(nn.Module):
    def __init__(self,num_g,hid):
        super().__init__()
        self.fc1=nn.Linear(num_g,200)
        #self.fc2=nn.Linear(200,hid)
         
    def forward(self,X):
        #h=torch.sigmoid(self.fc1(X))
        return self.fc1(X)
    
pt_auprc,pt_auroc=[],[]
ft_auprc,ft_auroc=[],[]

for d in tqdm(pretrain_name):
    if d in unlabelled:continue
    dataset=h5py.File('dataset/{n}.h5'.format(n=d),'r')
    with open('dataset/{n}.p'.format(n=d),'rb') as f:
        feature=pickle.load(f).toarray()   
        (num_c,num_g)=feature.shape
        print(num_c,num_g)
        enc=encoder(num_g,200)
        enc.load_state_dict(embed)
        enc.eval()   
        label=[]
        cell_class=dataset['obs']['cell_ontology_class']
        id2cell=list(set(list(cell_class)))
        cell2id={v:k for k, v in enumerate(id2cell)}
        for i in range(num_c):
            label.append(int(cell2id[cell_class[i]]))

    out=enc(torch.FloatTensor(feature))
    out=out.detach().numpy()
    pca=PCA(n_components=200)
    pca.fit(feature[:500])
    out2=pca.transform(feature)

    random_map=[i for i in range(num_c)]
    random.shuffle(random_map)

    feature1=np.array([out[random_map[i]] for i in range(num_c)])
    label1=np.array([label[random_map[i]] for i in range(num_c)])

    feature2=np.array([out2[random_map[i]] for i in range(num_c)])
    label2=np.array([label[random_map[i]] for i in range(num_c)])

    num_train=min(config.max_tr_num,int(num_c*config.per_train))

    feature_train1=feature1[:num_train]
    feature_test1=feature1[num_train:]
    label_train1=label1[:num_train]
    label_test1=label1[num_train:]

    feature_train2=feature2[:num_train]
    feature_test2=feature2[num_train:]
    label_train2=label2[:num_train]
    label_test2=label2[num_train:]

    if config.clf is 'rf':
        rfclf=RandomForestClassifier(max_depth=5,n_estimators=10,n_jobs=-1)
    elif config.clf is 'lr':
        rfclf=LogisticRegression(n_jobs=-1,max_iter=10000)
    elif config.clf is 'svm':
        rfclf=SVC(kernel='linear')
    else:
        raise AssertionError
    rfclf.fit(feature_train1,label_train1)
    a,b=(multi_auprc_auroc(rfclf.predict(feature_test1),label_test1,max(label)+1))

    if config.clf is 'rf':
        rfclf=RandomForestClassifier(max_depth=5,n_estimators=10,n_jobs=-1)
    elif config.clf is 'lr':
        rfclf=LogisticRegression(n_jobs=-1,max_iter=10000)
    elif config.clf is 'svm':
        rfclf=SVC(kernel='linear')
    else:
        raise AssertionError
    rfclf.fit(feature_train2,label_train2)
    c,d=(multi_auprc_auroc(rfclf.predict(feature_test2),label_test2,max(label)+1))

    print(a,b)
    print(c,d)

    pt_auprc.append(a)
    pt_auroc.append(b)
    ft_auprc.append(c)
    ft_auroc.append(d)

with open('svm.p','wb') as f:
    pickle.dump([pt_auprc,pt_auroc,ft_auprc,ft_auroc],f)

print(np.mean(pt_auprc),np.mean(pt_auroc))
print(np.mean(ft_auprc),np.mean(ft_auroc))
