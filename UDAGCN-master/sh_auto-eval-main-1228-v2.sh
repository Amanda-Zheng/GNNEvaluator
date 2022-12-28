CUDA_VISIBLE_DEVICES=1 python pretrain-gcn.py --seed=0 \
--lr 5e-3 --wd 5e-4 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16


