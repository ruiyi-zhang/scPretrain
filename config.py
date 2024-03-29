import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--mixed',action='store_true')
parser.add_argument('--is_pretrain',action='store_true')
parser.add_argument('--pt_epoch',type=int,default=50)
parser.add_argument('--ft_epoch',type=int, default=500)
parser.add_argument('--cuda',action='store_true')
parser.add_argument('--avg_cluster_num',type=int,default=100)
parser.add_argument('--batch_num',type=int,default=30)
parser.add_argument('--pt_lr',type=float,default=1e-4)
parser.add_argument('--ft_lr',type=float,default=0.002)
parser.add_argument('--hid',type=int,default=200)
parser.add_argument('--ft_batch_size',type=int,default=5000)
parser.add_argument('--per_train',type=float,default=0.6)
parser.add_argument('--per_val',type=float,default=0.1)
parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--pt_patience',type=int,default=5)
parser.add_argument('--max_tr_num',type=int,default=1000)
parser.add_argument('--kmeans',action='store_true')
parser.add_argument('--pretrain_output',action='store_true')
parser.add_argument('--pca_ft',action='store_true')
parser.add_argument('--pca_dim',type=int,default=500)
parser.add_argument('--pca_pt',action='store_true')
parser.add_argument('--pretrained_model',type=str,default='pretrain/scPretrain.out')
parser.add_argument('--save_model',type=str,default='pretrain/scPretrain.out')
parser.add_argument('--save_result',type=str,default='results/result.out')
parser.add_argument('--fold1',type=int,default=1)
parser.add_argument('--fold2',type=int,default=1)
parser.add_argument('--clf',type=str)
parser.add_argument('--np_input',type=str)
parser.add_argument('--ann_input',type=str)

args=parser.parse_args()
mixed=args.mixed
pt_epoch=args.pt_epoch
ft_epoch=args.ft_epoch
cuda=args.cuda
avg_cluster_num=args.avg_cluster_num
batch_num=args.batch_num
pt_lr=args.pt_lr
ft_lr=args.ft_lr
hid=args.hid
ft_batch_size=args.ft_batch_size
per_train=args.per_train
per_val=args.per_val
patience=args.patience
max_tr_num=args.max_tr_num
is_pretrain=args.is_pretrain
pt_patience=args.pt_patience
kmeans=args.kmeans
pretrain_output=args.pretrain_output
pca_ft=args.pca_ft
pca_pt=args.pca_pt
pca_dim=args.pca_dim
pretrained_model=args.pretrained_model
save_model=args.save_model
save_result=args.save_result
fold1=args.fold1
fold2=args.fold2
clf=args.clf
ann_input=args.ann_input
np_input=args.np_input

