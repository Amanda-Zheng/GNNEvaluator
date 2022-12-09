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

# test-demo-1: 观察source dataset train 和 valid 最初的mmd距离随着gcn的训练过程的变化
CUDA_VISIBLE_DEVICES=0 python UDAGCN_demo_xin_tmp.py --seed=0 \
--lr 3e-3 --epochs 300 --model GCN --full_s 1 --encoder_dim 16
#11/15 11:38:36 AM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 156, best_source_acc: 0.9635627269744873, best_target_acc: 0.7219433188438416
#11/15 11:38:36 AM Namespace(UDAGCN=False, encoder_dim=16, epochs=300, full_s=1, lr=0.003, model='GCN', name='UDAGCN', seed=0, source='acm', target='dblp')
#11/15 11:38:36 AM Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-1-0-20221115-113733-705307

# test-demo-2: 观察source/target dataset train 和 test 最初的mmd距离随着gcn的训练过程的变化
CUDA_VISIBLE_DEVICES=0 python UDAGCN_demo_xin_tmp.py --seed=0 \
--lr 3e-3 --epochs 300 --model GCN --full_s 1 --encoder_dim 16
#11/15 12:07:19 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 163, best_source_acc: 0.9676113128662109, best_target_acc: 0.7226604223251343
#11/15 12:07:19 PM Namespace(UDAGCN=False, encoder_dim=16, epochs=300, full_s=1, lr=0.003, model='GCN', name='UDAGCN', seed=0, source='acm', target='dblp')
#11/15 12:07:19 PM Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-1-0-20221115-120252-359495

# test-demo-3: 观察source/target dataset train 和 test 最初的mmd距离随着gcn的训练过程的变化
CUDA_VISIBLE_DEVICES=0 python UDAGCN_demo_xin_tmp.py --seed=0 \
--lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16
#11/15 01:55:21 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 26, best_source_acc: 0.8468285799026489, best_target_acc: 0.7149515748023987
#11/15 01:55:21 PM Namespace(UDAGCN=False, encoder_dim=16, epochs=300, full_s=0, lr=0.005, model='GCN', name='UDAGCN', seed=0, source='acm', target='dblp')
#11/15 01:55:21 PM Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-20221115-135151-313113

# test-demo-4: 观察source/target dataset train 和 test 最初的mmd距离随着gcn的训练过程的变化, 绘制target acc curve
CUDA_VISIBLE_DEVICES=0 python UDAGCN_demo_xin_tmp.py --seed=0 \
--lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16
11/15 02:22:53 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 26, best_source_acc: 0.8468285799026489, best_target_acc: 0.7149515748023987
11/15 02:22:53 PM Namespace(UDAGCN=False, encoder_dim=16, epochs=300, full_s=0, lr=0.005, model='GCN', name='UDAGCN', seed=0, source='acm', target='dblp')
11/15 02:22:53 PM Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-20221115-141922-840800

CUDA_VISIBLE_DEVICES=0 python auto-eval-main.py --seed=0 \
--lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16 --edge_drop_all_p 0.1
./logs/acm-to-dblp-GCN-full-0-0-20221209-212631-998242

CUDA_VISIBLE_DEVICES=0 python auto-eval-main.py --seed=0 \
--lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16 --edge_drop_all_p 0.6
this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-20221209-213413-345509


