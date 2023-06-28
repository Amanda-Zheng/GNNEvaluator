#M-GCN-CtAD
CUDA_VISIBLE_DEVICES=3 python pretrain_gnn-0.py --seed=0 --source network --target acm --target2 dblp \
--lr=1e-2 --wd=1e-5 --epochs 300 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16 --num_layers=2
#M-SAGE-CtAD
CUDA_VISIBLE_DEVICES=3 python pretrain_gnn-0.py --seed=0 --source network --target acm --target2 dblp \
--lr=5e-3 --wd=1e-6 --epochs 300 --model SAGE --full_s 0 --hid_dim 128 --encoder_dim 16 --num_layers=2
#M-GAT-CtAD
CUDA_VISIBLE_DEVICES=3 python pretrain_gnn-0.py --seed=0 --source network --target acm --target2 dblp \
--lr=0.005 --wd=1e-6 --epochs 300 --model GAT --full_s 0 --hid_dim 128 --encoder_dim 16 --num_layers=2
#M-GIN-CtAD
CUDA_VISIBLE_DEVICES=3 python pretrain_gnn-0.py --seed=0 --source network --target acm --target2 dblp \
--lr=0.01 --wd=1e-6 --epochs 300 --model GIN --full_s 0 --hid_dim 128 --encoder_dim 16 --num_layers=2
#M-MLP-CtAD
CUDA_VISIBLE_DEVICES=3 python pretrain_gnn-0.py --seed=0 --source network --target acm --target2 dblp \
--lr=0.001 --wd=1e-5 --epochs 300 --model MLP --full_s 0 --hid_dim 128 --encoder_dim 16 --num_layers=2