for seed in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=2 python UDAGCN_demo_xin.py --seed=${seed} \
--lr 1e-3 --epochs 600 --model SAGE --full_s 1 --encoder_dim 128
done
for seed in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=2 python UDAGCN_demo_xin.py --seed=${seed} \
--lr 1e-4 --epochs 600 --model SAGE --full_s 1 --encoder_dim 128
done
for seed in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=2 python UDAGCN_demo_xin.py --seed=${seed} \
--lr 1e-3 --epochs 600 --model SAGE --full_s 0 --encoder_dim 128
done
for seed in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=2 python UDAGCN_demo_xin.py --seed=${seed} \
--lr 1e-4 --epochs 600 --model SAGE --full_s 0 --encoder_dim 128
done