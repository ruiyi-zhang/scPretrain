import config
import torch.optim as optim
import numpy as np
import torch

from data import get_finetune_loader,dataset_name,get_gene_map
from model import clf,encoder
from metrics import accuracy_score,multi_auprc_auroc,kappa
from copy import deepcopy
from torch.nn import Parameter

def get_ft_embed(embed,pt_gene_list,ft_gene_list,is_human,gene_map=None):
    num_g_pt=len(pt_gene_list)
    num_g_ft=len(ft_gene_list)
    if config.pca_ft:
        embed_ft=encoder(config.pca_dim,config.hid)
    else:
        embed_ft=encoder(num_g_ft,config.hid)
    cnt=0
    temp=embed_ft.fc1.weight.cpu().detach().numpy()
    if config.pca_ft:
        embed_ft.fc1.weight=Parameter(embed['fc1.weight'])
    else:
        for k,g in enumerate(ft_gene_list):
            if is_human and g in gene_map.keys():
                g_new=gene_map[g]
            else:
                g_new=g
            if g_new in pt_gene_list:
                cnt+=1
                idx=pt_gene_list.index(g_new)
                temp[:,k]=embed['fc1.weight'][:,idx].cpu().detach().numpy()
    #print('pretrained_genes:'+str(cnt))
        embed_ft.fc1.weight=Parameter(torch.FloatTensor(temp))
    embed_ft.fc1.bias=Parameter(embed['fc1.bias'])
    #embed_ft.fc2.weight=Parameter(embed['fc2.weight'])
    #embed_ft.fc2.bias=Parameter(embed['fc2.bias'])
    return embed_ft

def finetune(dataset,loader_tr,loader_val,loader_tst,label_num,num_g,ft_gene_list,embed=None,pt_gene_list=None):
    
    if dataset in dataset_name['human']:
        is_human=True
        gene_map=get_gene_map()
    else:
        gene_map=None
        is_human=False
    if embed is not None:
        embed_ft=get_ft_embed(embed,pt_gene_list,ft_gene_list,is_human,gene_map)
    else:
        embed_ft=None
    model=clf(embed_ft,num_g,config.hid,label_num)
    if config.cuda:
        model=model.cuda()
    optimizer=optim.Adam(model.parameters(),lr=config.ft_lr)
    
    best_acc=0
    cnt_p=config.patience
    for e in range(config.ft_epoch):
        model.train()
        for (feature,label) in loader_tr:
            loss,out,_=model(feature,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        acc,num=0,0
        for (feature,label) in loader_val:
            loss,out,_=model(feature,label)
            acc+=accuracy_score(out.cpu().detach().numpy(),label.cpu().detach().numpy())*len(label)
            num+=len(label)
        acc/=num
        if acc>best_acc:
            best_model=deepcopy(model)
            cnt_p=config.patience
            best_acc=acc
            #print(e,best_acc)
        else:
            cnt_p-=1
            if cnt_p is 0:break
    
    model=best_model.eval()
    labels=np.array([])
    outs=np.array([])
    confs=np.array([])
    for k,(feature,label) in enumerate(loader_tst):
        loss,out,conf=model(feature,label)
        
        outs=np.concatenate((outs,out.cpu().detach().numpy())).astype('int')
        labels=np.concatenate((labels,label.cpu().detach().numpy())).astype('int')
        if k is 0:
            confs=conf.cpu().detach().numpy()
        else:
            confs=np.concatenate((confs,conf.cpu().detach().numpy()))
    return accuracy_score(outs,labels),kappa(outs,labels,label_num),multi_auprc_auroc(confs,labels,label_num)