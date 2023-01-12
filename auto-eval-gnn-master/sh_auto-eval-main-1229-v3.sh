CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 5e-3 --wd 1e-5 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
#12/29 11:19:26 AM Best Epoch: 34, best_source_test_acc: 0.8394061923027039, best_source_val_acc: 0.8515519499778748, best_target_acc: 0.6636787056922913
#12/29 11:19:26 AM Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.005, model='GCN', seed=0, source='acm', target='dblp', wd=1e-05)
#12/29 11:19:26 AM Finish!, this is the log dir = ./logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558

CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 3e-3 --wd 1e-5 --epochs 300 --model SAGE --full_s 0 --hid_dim 128 --encoder_dim 16
#./logs/acm-to-dblp-SAGE-full-0-0-20221229-115452-301677

CUDA_VISIBLE_DEVICES=1 python meta_set_save.py --seed=0 --source acm --num_metas 300
#./logs/Meta-save-acm-num-300-0-20221229-142336-945310

CUDA_VISIBLE_DEVICES=1 python meta_feat_acc.py --seed=0 --source acm \
--num_metas 300 --model GCN --hid_dim 128 --encoder_dim 16 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--aug_data_path './logs/Meta-save-acm-num-300-0-20221229-142336-945310/'
#./logs/Meta-feat-acc-acm-GCN-num-300-0-20221229-144816-935468

CUDA_VISIBLE_DEVICES=1 python meta_feat_acc.py --seed=0 --source acm \
--num_metas 300 --model SAGE --hid_dim 128 --encoder_dim 16 \
--model_path './logs/acm-to-dblp-SAGE-full-0-0-20221229-115452-301677/' \
--aug_data_path './logs/Meta-save-acm-num-300-0-20221229-142336-945310/'
#./logs/Meta-feat-acc-acm-SAGE-num-300-0-20221229-145216-765471

CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--load_path './logs/Meta-feat-acc-acm-GCN-num-300-0-20221229-144816-935468/'
#./logs/metaLR-acm-to-dblp-GCN-0-20221229-145546-042027

CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model SAGE --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-to-dblp-SAGE-full-0-0-20221229-115452-301677/' \
--load_path './logs/Meta-feat-acc-acm-SAGE-num-300-0-20221229-145216-765471'
#./logs/metaLR-acm-to-dblp-SAGE-0-20221229-145752-080724