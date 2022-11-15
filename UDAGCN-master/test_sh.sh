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

# test-demo-1: 观察train 和 valid 最初的mmd距离随着gcn的训练过程的变化
CUDA_VISIBLE_DEVICES=2 python UDAGCN_demo_xin_tmp.py --seed=0 \
--lr 3e-3 --epochs 300 --model GCN --full_s 1 --encoder_dim 16
#11/15 11:38:36 AM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 156, best_source_acc: 0.9635627269744873, best_target_acc: 0.7219433188438416
#11/15 11:38:36 AM Namespace(UDAGCN=False, encoder_dim=16, epochs=300, full_s=1, lr=0.003, model='GCN', name='UDAGCN', seed=0, source='acm', target='dblp')
#11/15 11:38:36 AM Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-1-0-20221115-113733-705307