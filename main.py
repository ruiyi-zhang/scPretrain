import torch
import config

from data import dataset_name,get_gene_list,unlabelled,get_finetune_loader
from tqdm import tqdm
from finetune import finetune
from pretrain import pretrain

pretrain_name=(dataset_name['mouse']+dataset_name['human'])
gene_list=get_gene_list(None)

if config.is_pretrain:
    if config.pretrain_output:
        with open(config.pretrained_model,'rb') as f:
            embed=torch.load(f)
    else:
        embed=None
    encoder,gene_list=pretrain(pretrain_name,False,embed)
    with open(config.save_model,'wb') as f:
        torch.save(encoder.state_dict(),f) 
else:
    with open(config.pretrained_model,'rb') as f:
        pretrain=torch.load(f)
    for ft_name in tqdm(dataset_name['mouse']+dataset_name['human']):
        if ft_name in unlabelled:continue
        with open(config.save_result,'a') as f:
            f.write(ft_name+'\n')
            loader_tr,loader_val,loader_tst,label_num,num_g,ft_gene_list=get_finetune_loader(ft_name,True)
            loader_tr2,loader_val2,loader_tst2,label_num2,num_g2,ft_gene_list2=get_finetune_loader(ft_name,False)
            f.write('scPretrain:'+str(finetune(ft_name,loader_tr,loader_val,loader_tst,label_num,num_g,ft_gene_list,pretrain,gene_list))+'\n')
            f.write('Without pretraining:'+str(finetune(ft_name,loader_tr2,loader_val2,loader_tst2,label_num2,num_g2,ft_gene_list2))+'\n')

