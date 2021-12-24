import torch
import h5py
import faiss
import numpy as np
import os
import pickle
import config
import random

from urllib.request import urlretrieve
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

dataset_name={
    'human':[
    'Nguyen_10x','Velten_Smart-seq2','Wang_Kidney','Young',
    'Guo','Lake_2018','Muraro','Enge','Philippeos','Vento-Tormo_Smart-seq2',
    'Wu_human',
    'Zheng','Baron_human','Hochane','Vento-Tormo_10x'
    ],
    'mouse':[
    'Quake_10x_Bladder','Quake_Smart-seq2' ,'Quake_10x',
    'Quake_Smart-seq2_Diaphragm',
    'Plasschaert','Quake_Smart-seq2_Large_Intestine',
    'Adam','Quake_10x_Lung',
    'Quake_Smart-seq2_Mammary_Gland','Quake_10x_Tongue','Quake_Smart-seq2_Spleen',
    'Haber_10x_largecell','Quake_10x_Kidney',
    'Quake_Smart-seq2_Bladder','Baron_mouse',
    'Quake_Smart-seq2_Brain_Non-Myeloid','Tusi',
    'Chen','Quake_Smart-seq2_Heart','Dahlin_10x',
    'Giraddi_10x','Quake_10x_Liver','Quake_Smart-seq2_Lung','Quake_Smart-seq2_Skin',
    'Haber_10x_FAE','Quake_10x_Heart_and_Aorta',
    'Quake_10x_Spleen','Shekhar','Green',
    'Karaiskos_mouse','Quake_Smart-seq2_Brain_Myeloid',
    'Montoro_10x','Park','Quake_Smart-seq2_Fat','Dahlin_mutant',
    'Qiu','Quake_Smart-seq2_Limb_Muscle','Wang_Lung',
    'Zeisel_2018','Haber_10x','Quake_10x_Bone_Marrow',
    'Quake_10x_Mammary_Gland','Quake_Smart-seq2_Pancreas','Quake_10x_Trachea',
    'Quake_Smart-seq2_Trachea','Bach',
    'Haber_10x_region','Quake_10x_Limb_Muscle','Quake_10x_Limb_Muscle','Quake_Smart-seq2_Bone_Marrow',
    'Campbell','Macosko'
    ]
}

unlabelled=[
    'Velten_Smart-seq2','Dahlin_10x','Tusi','Philippeos','Dahlin_mutant','Muraro','Philippeos'
    ,'Quake_10x_Heart_and_Aorta','Singh'
]

def download_datasets():
    for name in tqdm(dataset_name['human']):
        if not os.path.exists('dataset/{n}.h5'.format(n=name)):
            urlretrieve('https://cblast.gao-lab.org/{n}/{n}.h5'.format(n=name),'dataset/{n}.h5'.format(n=name))

    for name in tqdm(dataset_name['mouse']):
        if not os.path.exists('dataset/{n}.h5'.format(n=name)):
            urlretrieve('https://cblast.gao-lab.org/{n}/{n}.h5'.format(n=name),'dataset/{n}.h5'.format(n=name))

def get_gene_map():
    with open('dataset/ncbi_mgi_ensembl__mouse-lemur_human_mouse__orthologs__gene_names__one2one.csv') as f:
        gene_map={}
        for k,l in enumerate(f.readlines()):
            line=l.strip().split(',')
            if k is not 0:
                gene_map[line[1]]=line[2]
    return gene_map

def get_pretrain_data(name,gene_list):
    if name in dataset_name['human']:
        is_human=True
    else:
        is_human=False

    if os.path.exists('dataset/{n}.p'.format(n=name)):
        with open('dataset/{n}.p'.format(n=name),'rb') as f:
            feature_new=pickle.load(f).toarray().astype('float32')
            return feature_new
    f=h5py.File('dataset/{n}.h5'.format(n=name),'r')
    feature=csr_matrix((f['exprs']['data'],f['exprs']['indices'],f['exprs']['indptr']),
                shape=f['exprs']['shape']).toarray().astype('float32')
    var_list=[]
    gene_map=get_gene_map()
    for g in f['var_names']:
        if is_human:
            if str(g)[2:-1].upper() in gene_map.keys():
                var_list.append(gene_map[str(g)[2:-1].upper()])
            else:
                var_list.append(str(g)[2:-1].upper())
        else:
            var_list.append(str(g)[2:-1].upper())
    num_c,num_g=feature.shape
    num_g_new=len(gene_list)
    feature_new=np.zeros((num_c,num_g_new))
    for k,v in enumerate(gene_list):
        if v in var_list:
            feature_new[:,k]=feature[:,var_list.index(v)]
    with open('dataset/{n}.p'.format(n=name),'wb') as f:
        pickle.dump(csr_matrix(feature_new),f)
    return feature_new.astype('float32')

def get_gene_list(data_list):
    if os.path.exists('dataset/gene_lst.p'):
        with open('dataset/gene_lst.p','rb') as f:
            gene_list=pickle.load(f)
        return gene_list
    else:
        gene_map=get_gene_map()
        for k,v in enumerate(dataset_name['mouse']):
            if k is 0:
                genes_m=set(h5py.File(v+'.h5','r')['var_names'])
            else:
                genes_m=genes_m|set(h5py.File(v+'.h5','r')['var_names'])
        for k,v in enumerate(dataset_name['human']):
            if k is 0:
                genes=set(h5py.File(v+'.h5','r')['var_names'])
            else:
                genes=genes|set(h5py.File(v+'.h5','r')['var_names'])
        human_genes=[]
        mouse_genes=[]
        cnt=0
        for g in genes:
            gene=str(g)[2:-1].upper()
            if gene in gene_map.keys():
                human_genes.append(gene_map[gene])
                cnt+=1
            else:
                human_genes.append(gene)
        
        for g in genes_m:
            gene=str(g)[2:-1].upper()
            mouse_genes.append(gene)
        gene_lst=list(set(human_genes)&set(mouse_genes))
        with open('dataset/gene_lst.p','wb') as f:
            pickle.dump(gene_lst,f)    
    return gene_list

def run_kmeans(x, nmb_clusters):
    n_data, d = x.shape
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    index=faiss.IndexFlatL2(d)
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]

def get_pretrain_label(name,feature):
    if os.path.exists('dataset/{n}_cluster_3.p'.format(n=name)) and not config.pretrain_output:
        with open('dataset/{n}_cluster_3.p'.format(n=name),'rb') as f:
            label=pickle.load(f)
    elif config.kmeans:
        label1=run_kmeans(feature,len(feature)//config.avg_cluster_num)
        label2=run_kmeans(feature,len(feature)//(config.avg_cluster_num//2))
        label3=run_kmeans(feature,len(feature)//(config.avg_cluster_num*2)+1)
        label=np.stack((label1,label2,label3),axis=-1)
        if not config.pretrain_output:
            with open('dataset/{n}_cluster_3.p'.format(n=name),'wb') as f:
                pickle.dump(label,f)
    else:
        cluster=AgglomerativeClustering(len(feature)//config.avg_cluster_num).fit(feature)
        label=cluster.labels_
        with open('dataset/{n}_cluster_hi.p'.format(n=name),'wb') as f:
            pickle.dump(label,f)
    return label

class dataset(Dataset):
    def __init__(self,feature,label):
        self.feature=feature
        self.label=label
    
    def __getitem__(self,idx):
        return (self.feature[idx],self.label[idx])

    def __len__(self):
        return len(self.label)

def collate(batch):
    feature=[]
    label=[]
    for f,l in batch:
        feature.append(f)
        label.append(l)
    
    if config.pca_ft or config.pca_pt:
        feature=torch.FloatTensor(np.array(feature))
    else:
        feature=torch.FloatTensor(np.array(feature)).log1p()
    label=torch.LongTensor(np.array(label))

    if config.cuda:
        feature=feature.cuda()
        label=label.cuda()

    return feature,label

def get_pretrain_loader(name_list,mix,embed=None):
    gene_list=get_gene_list(name_list)
    loader_list=[]
    label_num_list=[]
    for n in name_list:
        if n in unlabelled:continue
        #print(n)
        feature=get_pretrain_data(n,gene_list)
        (num_c,num_g)=feature.shape
        num_train=min(int(num_c*config.per_train),config.max_tr_num)
        num_val=min(int(num_c*config.per_val),config.max_tr_num)
        if config.pca_pt:
            #dim=min(config.pca_dim,feature.shape[1]//2,feature.shape[0]//2)
            pca=PCA(n_components=config.pca_dim)
            pca.fit(feature[:num_train+num_val])
            feature=pca.transform(feature)
        if embed is not None:
            embed.eval()
            if config.pca_pt:
                feature_out=embed(torch.FloatTensor(np.array(feature)))
            else:
                feature_out=embed(torch.FloatTensor(np.array(feature)).log1p())
            feature_out=feature_out.detach().numpy()
            label=get_pretrain_label(n,feature_out)
        else:
            label=get_pretrain_label(n,feature)
        pt_dataset=dataset(feature,label)
        loader=DataLoader(
            dataset=pt_dataset,
            batch_size=len(pt_dataset)//config.batch_num,
            collate_fn=collate,
            shuffle=True
        )
        loader_list.append(loader)
        label_num_list.append(label.max(axis=0)+1)

    return loader_list,label_num_list

def get_finetune_data(name):
    f=h5py.File('dataset/{n}.h5'.format(n=name),'r')
    feature=csr_matrix((f['exprs']['data'],f['exprs']['indices'],f['exprs']['indptr']),
                shape=f['exprs']['shape']).toarray()
    (num_c,num_g)=feature.shape

    label=[]
    cell_class=f['obs']['cell_ontology_class']
    id2cell=list(set(list(cell_class)))
    cell2id={v:k for k, v in enumerate(id2cell)}

    for i in range(num_c):
        label.append(int(cell2id[cell_class[i]]))

    gene_list=[str(g)[2:-1].upper() for g in f['var_names']]

    return feature,label,gene_list

def get_finetune_loader(name,pretrained=True):
    if config.np_input:
        feature=np.load(config.np_input+'feature.npy')
        label=np.load(config.np_input+'label.npy')
        gene_list=np.load(config.np_input+'gene_list.npy').tolist()
    elif config.ann_input:
        import anndata
        adata=anndata.read_h5ad(config.ann_input)
        feature=adata.X
        label=adata.obs.to_numpy()
        gene_list=adata.var.values.tolist()
    else:
        feature,label,gene_list=get_finetune_data(name)
    (num_c,num_g)=feature.shape
    num_train=min(int(num_c*config.per_train),config.max_tr_num)
    num_val=min(int(num_c*config.per_val),config.max_tr_num)

    if config.pca_ft:
        pca=PCA(n_components=config.pca_dim)
        pca.fit(feature[:num_train+num_val])
        feature=pca.transform(feature)
    
    #print(feature.shape)
    #print(np.isnan(feature).sum())
    random_map=[i for i in range(num_c)]
    random.shuffle(random_map)

    feature=np.array([feature[random_map[i]] for i in range(num_c)])
    label=np.array([label[random_map[i]] for i in range(num_c)])

    

    dataset_tr=dataset(feature[:num_train],label[:num_train])
    dataset_val=dataset(feature[num_train:num_train+num_val],label[num_train:num_train+num_val])
    dataset_tst=dataset(feature[num_train+num_val:],label[num_train+num_val:])
    loader_tr=DataLoader(
        dataset=dataset_tr,
        batch_size=config.ft_batch_size,
        collate_fn=collate,
        shuffle=True
    )
    loader_val=DataLoader(
        dataset=dataset_val,
        batch_size=config.ft_batch_size,
        collate_fn=collate,
        shuffle=True
    )
    loader_tst=DataLoader(
        dataset=dataset_tst,
        batch_size=config.ft_batch_size,
        collate_fn=collate,
        shuffle=True
    )
    label_num=max(label)+1
    return loader_tr,loader_val,loader_tst,label_num,num_g,gene_list
