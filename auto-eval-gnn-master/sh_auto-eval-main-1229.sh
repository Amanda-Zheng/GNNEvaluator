CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 5e-3 --wd 1e-5 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
12/29 11:19:26 AM Best Epoch: 34, best_source_test_acc: 0.8394061923027039, best_source_val_acc: 0.8515519499778748, best_target_acc: 0.6636787056922913
12/29 11:19:26 AM Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.005, model='GCN', seed=0, source='acm', target='dblp', wd=1e-05)
12/29 11:19:26 AM Finish!, this is the log dir = ./logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558

CUDA_VISIBLE_DEVICES=1 python meta_train_test.py --seed=0 \
--num_metas 300 --model GCN --hid_dim 128 --encoder_dim 16 \
--save_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558'
12/29 11:39:39 AM Namespace(aug_method='node_mix', edge_drop_all_p=0.8532459761585128, encoder_dim=16, hid_dim=128, mix_lamb=0.9692058717176852, model='GCN', node_drop_val_p=0.05, node_fmask_all_p=0.4753247822120824, num_metas=300, save_path='./logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558', seed=0, source='acm')
12/29 11:39:39 AM Finish, this is the log dir = ./logs/Meta-trate-acm-GCN-num-300-0-20221229-113920-934071

CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--load_path './logs/Meta-trate-acm-GCN-num-300-0-20221229-113920-934071/'
<<!
CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 5e-3 --wd 5e-4 --epochs 300 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
12/29 11:16:36 AM Best Epoch: 45, best_source_test_acc: 0.8367071151733398, best_source_val_acc: 0.8515519499778748, best_target_acc: 0.6588382720947266
12/29 11:16:36 AM Namespace(encoder_dim=16, epochs=300, full_s=0, hid_dim=128, lr=0.005, model='GCN', seed=0, source='acm', target='dblp', wd=0.0005)
12/29 11:16:36 AM Finish!, this is the log dir = ./logs/acm-to-dblp-GCN-full-0-0-20221229-111628-790702

CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 3e-3 --wd 5e-4 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
12/29 11:17:19 AM Best Epoch: 71, best_source_test_acc: 0.8326585292816162, best_source_val_acc: 0.8515519499778748, best_target_acc: 0.6484403014183044
12/29 11:17:19 AM Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.003, model='GCN', seed=0, source='acm', target='dblp', wd=0.0005)
12/29 11:17:19 AM Finish!, this is the log dir = ./logs/acm-to-dblp-GCN-full-0-0-20221229-111713-690205

CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 3e-3 --wd 1e-4 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
12/29 11:18:22 AM Best Epoch: 55, best_source_test_acc: 0.8353576064109802, best_source_val_acc: 0.8515519499778748, best_target_acc: 0.6265686750411987
12/29 11:18:22 AM Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.003, model='GCN', seed=0, source='acm', target='dblp', wd=0.0001)
12/29 11:18:22 AM Finish!, this is the log dir = ./logs/acm-to-dblp-GCN-full-0-0-20221229-111816-140687

CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 5e-3 --wd 1e-4 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
12/29 11:18:56 AM Best Epoch: 36, best_source_test_acc: 0.8319838047027588, best_source_val_acc: 0.8475033640861511, best_target_acc: 0.6622444987297058
12/29 11:18:56 AM Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.005, model='GCN', seed=0, source='acm', target='dblp', wd=0.0001)
12/29 11:18:56 AM Finish!, this is the log dir = ./logs/acm-to-dblp-GCN-full-0-0-20221229-111850-464356

CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 3e-3 --wd 1e-5 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
12/29 11:19:55 AM Best Epoch: 49, best_source_test_acc: 0.837381899356842, best_source_val_acc: 0.8488528728485107, best_target_acc: 0.6339189410209656
12/29 11:19:55 AM Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.003, model='GCN', seed=0, source='acm', target='dblp', wd=1e-05)
12/29 11:19:55 AM Finish!, this is the log dir = ./logs/acm-to-dblp-GCN-full-0-0-20221229-111949-696370

CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 1e-2 --wd 1e-5 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
12/29 11:20:32 AM Best Epoch: 25, best_source_test_acc: 0.827260434627533, best_source_val_acc: 0.8488528728485107, best_target_acc: 0.7163857817649841
12/29 11:20:32 AM Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.01, model='GCN', seed=0, source='acm', target='dblp', wd=1e-05)
12/29 11:20:32 AM Finish!, this is the log dir = ./logs/acm-to-dblp-GCN-full-0-0-20221229-112026-888862

CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 3e-2 --wd 1e-5 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
12/29 11:21:13 AM Best Epoch: 12, best_source_test_acc: 0.8360323905944824, best_source_val_acc: 0.8488528728485107, best_target_acc: 0.6159914135932922
12/29 11:21:13 AM Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.03, model='GCN', seed=0, source='acm', target='dblp', wd=1e-05)
12/29 11:21:13 AM Finish!, this is the log dir = ./logs/acm-to-dblp-GCN-full-0-0-20221229-112108-004579
!