import h5py
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import torch
import config
from sklearn.metrics import accuracy_score,silhouette_score,adjusted_rand_score
from data import dataset_name,unlabelled,run_kmeans

with open('pretrain_3clus_out19.pt','rb') as f:
    embed=torch.load(f)

sil=[]
for d in dataset_name['mouse']+dataset_name['human']:
    if d in unlabelled:continue
    f_test=h5py.File('dataset/{n}.h5'.format(n=d),'r')
    with open('dataset/{n}.p'.format(n=d),'rb') as f:
        clus=pickle.load(f)

    dataset=f_test

    with open('dataset/{n}.p'.format(n=d),'rb') as f:
        feature=pickle.load(f).toarray()    
    (num_c,num_g)=feature.shape
    print(d)
    label=[]
    cell_class=dataset['obs']['cell_ontology_class']
    id2cell=list(set(list(cell_class)))
    cell2id={v:k for k, v in enumerate(id2cell)}

    for i in range(num_c):
        label.append(int(cell2id[cell_class[i]]))
    
    from model import encoder
    enc=encoder(num_g,200)
    enc.load_state_dict(embed)
    enc.eval()
    temp=[]
    for i in range(10):
        embed_feature=enc(torch.FloatTensor(feature)).detach().numpy()
        #label_pre=run_kmeans(embed_feature.astype('float32'),len(embed_feature)*2//(config.avg_cluster_num*(2**i))+1)
        label_pre=run_kmeans(embed_feature.astype('float32'),max(label)+1)
        temp.append(adjusted_rand_score(label,label_pre))
    print(np.mean(temp))
    sil.append((temp))
with open('sil_pt.p','wb') as f:
        pickle.dump(sil,f)
#print(np.mean(sil))