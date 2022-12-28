<<!
对称性测试,dist_t_full_aug_val = dist_aug_val_t_full, 问题在于计算这个kernel非常消耗内存
CUDA_VISIBLE_DEVICES=1 python auto-eval-main.py --seed=0 \
--lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16 --aug_method 'node_mix' \
--mix_lamb=0.1
12/28 10:51:05 AM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 44, best_source_acc: 0.8238866329193115, best_target_acc: 0.6964861750602722, dist_s_tra_aug_val = 0.4819006323814392,dist_aug_val_t_full = 0, dist_t_full_aug_val = 0.39550161361694336, dist_s_val_t_full=0.7430355548858643
12/28 10:51:05 AM Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.1, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.05, seed=0, source='acm', target='dblp')
12/28 10:51:05 AM Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_mix-20221228-105054-627261
!
for edge_p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  for lamb in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
  do
    for node_fp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
    CUDA_VISIBLE_DEVICES=1 python auto-eval-main.py --seed=0 \
    --lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16 --aug_method 'combo' \
    --edge_drop_all_p=${edge_p} --mix_lamb=${lamb} --node_fmask_all_p=${node_fp}
    done
  done
done
CUDA_VISIBLE_DEVICES=1 python pretrain.py --seed=0 \
--lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16
12/28 03:28:17 PM source: acm, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 23, best_source_acc: 0.8481780886650085
12/28 03:28:17 PM Namespace(UDAGCN=False, encoder_dim=16, epochs=300, full_s=0, lr=0.005, model='GCN', name='UDAGCN', save_m_pt='checkpoints/', seed=0, source='acm')
12/28 03:28:17 PM Finish!, this is the log dir = ./logs/acm-GCN-full-0-0-20221228-152812-169114 and cpk dir = checkpoints/

CUDA_VISIBLE_DEVICES=1 python pretrain.py --seed=0 \
--lr 3e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16
12/28 03:34:48 PM source: acm, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 33, best_source_acc: 0.8522266745567322
12/28 03:34:48 PM Namespace(UDAGCN=False, encoder_dim=16, epochs=300, full_s=0, lr=0.003, model='GCN', name='UDAGCN', save_m_pt='checkpoints/', seed=0, source='acm')
12/28 03:34:48 PM Finish!, this is the log dir = ./logs/acm-GCN-full-0-0-20221228-153442-415490 and cpk dir = checkpoints/


CUDA_VISIBLE_DEVICES=1 python pretrain.py --seed=0 \
--lr 1e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16
12/28 03:32:19 PM source: acm, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 98, best_source_acc: 0.8502023816108704
12/28 03:32:19 PM Namespace(UDAGCN=False, encoder_dim=16, epochs=300, full_s=0, lr=0.001, model='GCN', name='UDAGCN', save_m_pt='checkpoints/', seed=0, source='acm')
12/28 03:32:19 PM Finish!, this is the log dir = ./logs/acm-GCN-full-0-0-20221228-153212-842026 and cpk dir = checkpoints/

CUDA_VISIBLE_DEVICES=1 python meta_train_test.py --seed=0 \
--num_metas 300 --model GCN --full_s 0 --encoder_dim 16 \
--save_path './logs/acm-GCN-full-0-0-20221228-153442-415490/checkpoints/'
12/28 05:13:56 PM The size of meta data: feat = (300, 1), acc = (300, 1)
12/28 05:13:56 PM Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.8532459761585128, encoder_dim=16, full_s=0, mix_lamb=0.9692058717176852, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.4753247822120824, num_metas=300, save_path='./logs/acm-GCN-full-0-0-20221228-153442-415490/checkpoints/', seed=0, source='acm')
12/28 05:13:56 PM Finish, this is the log dir = ./logs/acm-GCN-full-0-0-20221228-171336-369140

CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model GCN --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-GCN-full-0-0-20221228-153442-415490/checkpoints/' \
--load_path './logs/acm-GCN-full-0-0-20221228-171336-369140/'
12/28 08:22:11 PM Namespace(encoder_dim=16, load_path='./logs/acm-GCN-full-0-0-20221228-171336-369140/', model='GCN', model_path='./logs/acm-GCN-full-0-0-20221228-153442-415490/checkpoints/', seed=0, source='acm', target='dblp', val_num=30)
12/28 08:22:11 PM Finish, this is the log dir = ./logs/metaLR-acm-to-dblp-GCN-0-20221228-202203-329412

CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=1 \
--source acm --target dblp --model GCN --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-GCN-full-0-0-20221228-153442-415490/checkpoints/' \
--load_path './logs/acm-GCN-full-0-0-20221228-171336-369140/'

