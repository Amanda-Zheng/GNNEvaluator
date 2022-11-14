for seed in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=2 python UDAGCN_demo_xin.py --seed=${seed} \
--lr 3e-3 --epochs 300 --model GCN --full_s 1 --encoder_dim 16
done
for seed in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=2 python UDAGCN_demo_xin.py --seed=${seed} \
--lr 1e-2 --epochs 300 --model GCN --full_s 1 --encoder_dim 16
done
for seed in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=2 python UDAGCN_demo_xin.py --seed=${seed} \
--lr 3e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16
done
for seed in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=2 python UDAGCN_demo_xin.py --seed=${seed} \
--lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16
done