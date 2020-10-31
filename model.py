import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class encoder(nn.Module):
    def __init__(self,num_g,hid):
        super().__init__()
        self.fc1=nn.Linear(num_g,200)
        #self.fc2=nn.Linear(200,hid)
         
    def forward(self,X):
        #h=torch.sigmoid(self.fc1(X))
        return self.fc1(X)

class clf(nn.Module):
    def __init__(self,pt_encoder,num_g,hid,num_l):
        super().__init__()
        if pt_encoder is None:
            if config.pca_ft:
                self.encoder=encoder(config.pca_dim,hid)
            else:
                self.encoder=encoder(num_g,hid)
        else:
            self.encoder=pt_encoder
        self.fc1=nn.Linear(hid,hid//2)
        self.fc2=nn.Linear(hid//2,num_l)
    def forward(self,X,y):
        h=torch.sigmoid(self.encoder(X))
        h=torch.sigmoid(self.fc1(h))
        out=self.fc2(h)
        return F.cross_entropy(out,y),torch.argmax(F.softmax(out,dim=1),dim=1),F.softmax(out,dim=1)

class mtclf(nn.Module):
    def __init__(self,num_g,hid,num_cl,pt_encoder=None):
        super().__init__()
        self.len=len(num_cl)
        self.num_cl=np.array(num_cl)
        if pt_encoder is None:
            if config.pca_pt:
                self.encoder=encoder(config.pca_dim,hid)
            else:
                self.encoder=encoder(num_g,hid)
        else:
            self.encoder=pt_encoder
            self.encoder.train()
        self.mt=nn.ModuleList([nn.Linear(hid,n) for n in self.num_cl[:,0]])
        self.mt2=nn.ModuleList([nn.Linear(hid,n) for n in self.num_cl[:,1]])
        self.mt3=nn.ModuleList([nn.Linear(hid,n) for n in self.num_cl[:,2]])
        self.loss=nn.CrossEntropyLoss()
        

    def forward(self,X,y):
        h=[]
        out,out1,out2=[],[],[]

        for k,v in enumerate(range(self.len)):
            h.append(torch.sigmoid(self.encoder(X[k])))
            out.append(torch.sigmoid(self.mt[k](h[k])))
            out1.append(torch.sigmoid(self.mt2[k](h[k])))
            out2.append(torch.sigmoid(self.mt3[k](h[k])))
            if k is 0:
                l=self.loss(out[k],y[k][:,0])+self.loss(out1[k],y[k][:,1])+self.loss(out2[k],y[k][:,2])
            else:
                l=l+self.loss(out[k],y[k][:,0])+self.loss(out1[k],y[k][:,1])+self.loss(out2[k],y[k][:,2])
        return l