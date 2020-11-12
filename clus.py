import pickle
import h5py
import config
import torch.nn as nn
import torch
import umap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class encoder(nn.Module):
    def __init__(self,num_g,hid):
        super().__init__()
        self.fc1=nn.Linear(num_g,200)
        #self.fc2=nn.Linear(200,hid)
         
    def forward(self,X):
        #h=torch.sigmoid(self.fc1(X))
        return self.fc1(X)

with open('dataset/Quake_10x_Kidney.p','rb') as f:
        feature=pickle.load(f).toarray()
        
dataset=h5py.File('dataset/Quake_10x_Kidney.h5','r')
   
(num_c,num_g)=feature.shape
print(num_c,num_g)
    
label=[]
cell_class=dataset['obs']['cell_ontology_class']
id2cell=list(set(list(cell_class)))
cell2id={v:k for k, v in enumerate(id2cell)}

for i in range(num_c):
    label.append(int(cell2id[cell_class[i]]))


with open(config.pretrained_model,'rb') as f:
    embed=torch.load(f)

enc=encoder(num_g,200)
enc.load_state_dict(embed)
embed_feature=enc(torch.FloatTensor(feature)).detach().numpy()


reducer=umap.UMAP()
reducer.fit(feature[:1000])
#reducer=TSNE(n_jobs=-1)
feature_new=reducer.transform(feature)
reducer=umap.UMAP()
embed_new=reducer.fit_transform(embed_feature)

plt.figure(figsize=[15,15])
#plt.plot(x,x)
#plt.scatter(feature_tsne[:,0],feature_tsne[:,1],c=label,linewidths=(np.zeros(len(feature_tsne))+0.0001))
#color = matplotlib.rcParams["axes.prop_cycle"]
matplotlib.rcParams['pdf.fonttype'] = 42
for i in range(max(label)+1):
    idx=np.where(np.array(label)==i)[0]
    print(idx.shape)
    plt.scatter(embed_new[idx,0],embed_new[idx,1],
                linewidths=(np.zeros(len(idx))+1e-10),
                label=str(id2cell[i])[2].upper()+str(id2cell[i])[3:-1])
plt.title('scPretrain (Quake 10x Kidney)',fontsize=50)
plt.xticks([])
plt.yticks([])
font={'size':50}
plt.ylabel('UMAP 2',fontdict=font)
plt.xlabel('UMAP 1',fontdict=font)
plt.legend(fontsize=40,frameon=False,loc=3,bbox_to_anchor=(-0.1,1.1),ncol=1,markerscale=5)
plt.savefig('figs/kidney_pretrain.pdf',bbox_inches='tight')

plt.figure(figsize=[15,15])
#plt.plot(x,x)
#plt.scatter(feature_tsne[:,0],feature_tsne[:,1],c=label,linewidths=(np.zeros(len(feature_tsne))+0.0001))
matplotlib.rcParams['pdf.fonttype'] = 42
for i in range(max(label)+1):
    idx=np.where(np.array(label)==i)[0]
    print(idx.shape)
    plt.scatter(feature_new[idx,0],feature_new[idx,1],
                linewidths=(np.zeros(len(idx))+1e-10),
                label=str(id2cell[i])[2].upper()+str(id2cell[i])[3:-1])
plt.title('Without pre-training (Quake 10x Kidney)',fontsize=50)
plt.xticks([])
plt.yticks([]) 
font={'size':50}
plt.ylabel('UMAP 2',fontdict=font)
plt.xlabel('UMAP 1',fontdict=font)
plt.savefig('figs/kidney_finetune.pdf',bbox_inches='tight')