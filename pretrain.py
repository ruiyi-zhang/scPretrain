import numpy as np
import config
import torch.optim as optim
import os

from tqdm import tqdm
from data import get_pretrain_loader,dataset_name,get_gene_list
from sklearn.metrics import accuracy_score,silhouette_score

from model import mtclf,encoder



def clus_cnt(label_lst):
    n=max(label_lst)+1
    cnt_lst=[0 for i in range(n)]
    for label in label_lst:
        cnt_lst[label]+=1
    print(cnt_lst)

class list_loader:
    def __init__(self,loader_list):
        self.loader_list=[iter(l) for l in loader_list]
        self.cnt=len(self.loader_list[0])

    def __iter__(self):
        self.cnt-=1
        return self

    def __next__(self):
        if self.cnt>=0:
            feature_list=[]
            label_list=[]
            for loader in self.loader_list:
                (feature,label)=next(loader)
                feature_list.append(feature)
                label_list.append(label)
            return feature_list,label_list
        else:
            raise StopIteration


def pretrain(name_list,mix,embed=None):
    gene_list=get_gene_list(None)
    num_g=len(gene_list)
    if embed is None:
        loader_list,label_num_list=get_pretrain_loader(name_list,mix)
    else:
        if config.pca_pt:
            enc=encoder(config.pca_dim,config.hid)
        else:
            enc=encoder(num_g,config.hid)
        enc.load_state_dict(embed)
        loader_list,label_num_list=get_pretrain_loader(name_list,mix,enc)
    if embed is None:
        model=mtclf(num_g,config.hid,label_num_list)
    else:
        model=mtclf(num_g,config.hid,label_num_list,enc)
    if config.cuda:
        model=model.cuda()
    optimizer=optim.Adam(model.parameters(),lr=config.pt_lr)

    cnt=config.pt_patience
    min_loss=1000

    for e in range(config.pt_epoch):
        total=0
        for (feature_list, label_list) in list_loader(loader_list):
            loss=model(feature_list,label_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total+=float(loss)
        
        total/=len(loader_list)
        print(e,total)
        if total<min_loss:
            min_loss=total
            cnt=config.pt_patience
        else:
            cnt-=1
            if cnt==0:break

    return model.encoder,gene_list

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    