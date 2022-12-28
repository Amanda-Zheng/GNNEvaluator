CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 5e-3 --wd 5e-4 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
12/28 09:38:40 PM Epoch: 26, best_source_acc: 0.8441295027732849
12/28 09:38:40 PM Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.005, model='GCN', seed=0, source='acm', wd=0.0005)
12/28 09:38:40 PM Finish!, this is the log dir = ./logs/acm-GCN-full-0-0-20221228-213830-683591

CUDA_VISIBLE_DEVICES=1 python meta_train_test.py --seed=0 \
--num_metas 300 --model GCN --hid_dim 128 --encoder_dim 16 \
--save_path './logs/acm-GCN-full-0-0-20221228-213830-683591/'
12/28 11:01:15 PM Namespace(aug_method='node_mix', edge_drop_all_p=0.8532459761585128, encoder_dim=16, hid_dim=128, mix_lamb=0.9692058717176852, model='GCN', node_drop_val_p=0.05, node_fmask_all_p=0.4753247822120824, num_metas=300, save_path='./logs/acm-GCN-full-0-0-20221228-213830-683591/', seed=0, source='acm')
12/28 11:01:15 PM Finish, this is the log dir = ./logs/Meta-trate-acm-GCN-num-300-0-20221228-230057-132889

CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-GCN-full-0-0-20221228-213830-683591/' \
--load_path './logs/Meta-trate-acm-GCN-num-300-0-20221228-230057-132889/'
./logs/metaLR-acm-to-dblp-GCN-0-20221228-231138-915766
Compare: LR target acc = [[0.6521487]] vs. Real target acc = 0.6423449516296387

CUDA_VISIBLE_DEVICES=1 python meta_train_test.py --seed=0 \
--num_metas 1000 --model GCN --hid_dim 128 --encoder_dim 16 \
--save_path './logs/acm-GCN-full-0-0-20221228-213830-683591/'
12/28 11:13:52 PM Namespace(aug_method='combo', edge_drop_all_p=0.19970392819903238, encoder_dim=16, hid_dim=128, mix_lamb=0.5414662556089532, model='GCN', node_drop_val_p=0.05, node_fmask_all_p=0.3680839022202911, num_metas=1000, save_path='./logs/acm-GCN-full-0-0-20221228-213830-683591/', seed=0, source='acm')
12/28 11:13:52 PM Finish, this is the log dir = ./logs/Meta-trate-acm-GCN-num-1000-0-20221228-231256-798274

CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-GCN-full-0-0-20221228-213830-683591/' \
--load_path './logs/Meta-trate-acm-GCN-num-1000-0-20221228-231256-798274/'
12/28 11:14:55 PM Compare: LR target acc = [[0.6586882]] vs. Real target acc = 0.6423449516296387
12/28 11:14:55 PM Finish, this is the log dir = ./logs/metaLR-acm-to-dblp-GCN-0-20221228-231452-171494

CUDA_VISIBLE_DEVICES=1 python pretrain-gcn.py --seed=0 \
--lr 3e-3 --wd 1e-4 --epochs 200 --model SAGE --full_s 0 --hid_dim 128 --encoder_dim 16
12/28 11:23:17 PM Epoch: 30, best_source_acc: 0.8360323905944824
12/28 11:23:17 PM Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.003, model='SAGE', seed=0, source='acm', wd=0.0001)
12/28 11:23:17 PM Finish!, this is the log dir = ./logs/acm-SAGE-full-0-0-20221228-232308-116666

CUDA_VISIBLE_DEVICES=1 python meta_train_test.py --seed=0 \
--num_metas 300 --model SAGE --hid_dim 128 --encoder_dim 16 \
--save_path './logs/acm-SAGE-full-0-0-20221228-232308-116666/'
12/28 11:24:22 PM Namespace(aug_method='node_mix', edge_drop_all_p=0.8532459761585128, encoder_dim=16, hid_dim=128, mix_lamb=0.9692058717176852, model='SAGE', node_drop_val_p=0.05, node_fmask_all_p=0.4753247822120824, num_metas=300, save_path='./logs/acm-SAGE-full-0-0-20221228-232308-116666/', seed=0, source='acm')
12/28 11:24:22 PM Finish, this is the log dir = ./logs/Meta-trate-acm-SAGE-num-300-0-20221228-232359-382207

CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model SAGE --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-SAGE-full-0-0-20221228-232308-116666/' \
--load_path './logs/Meta-trate-acm-SAGE-num-300-0-20221228-232359-382207/'
12/28 11:25:09 PM Compare: LR target acc = [[0.69140404]] vs. Real target acc = 0.7158479690551758
12/28 11:25:09 PM Namespace(encoder_dim=16, hid_dim=128, load_path='./logs/Meta-trate-acm-SAGE-num-300-0-20221228-232359-382207/', model='SAGE', model_path='./logs/acm-SAGE-full-0-0-20221228-232308-116666/', seed=0, source='acm', target='dblp', val_num=30)
12/28 11:25:09 PM Finish, this is the log dir = ./logs/metaLR-acm-to-dblp-SAGE-0-20221228-232506-570468
Test set: R2 :0.1347 RMSE: 0.1444 MAE: 0.1049

CUDA_VISIBLE_DEVICES=1 python meta_train_test.py --seed=0 \
--num_metas 1000 --model SAGE --hid_dim 128 --encoder_dim 16 \
--save_path './logs/acm-SAGE-full-0-0-20221228-232308-116666/'
12/28 11:27:03 PM Namespace(aug_method='combo', edge_drop_all_p=0.19970392819903238, encoder_dim=16, hid_dim=128, mix_lamb=0.5414662556089532, model='SAGE', node_drop_val_p=0.05, node_fmask_all_p=0.3680839022202911, num_metas=1000, save_path='./logs/acm-SAGE-full-0-0-20221228-232308-116666/', seed=0, source='acm')
12/28 11:27:03 PM Finish, this is the log dir = ./logs/Meta-trate-acm-SAGE-num-1000-0-20221228-232552-472125


CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model SAGE --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-SAGE-full-0-0-20221228-232308-116666/' \
--load_path './logs/Meta-trate-acm-SAGE-num-1000-0-20221228-232552-472125/'
12/28 11:28:07 PM Finish, this is the log dir = ./logs/metaLR-acm-to-dblp-SAGE-0-20221228-232803-926476
Compare: LR target acc = [[0.6995973]] vs. Real target acc = 0.7158479690551758
Test set: R2 :0.1120 RMSE: 0.1463 MAE: 0.1023

CUDA_VISIBLE_DEVICES=1 python meta_train_test.py --seed=0 \
--num_metas 2000 --model SAGE --hid_dim 128 --encoder_dim 16 \
--save_path './logs/acm-SAGE-full-0-0-20221228-232308-116666/'
Finish, this is the log dir = ./logs/Meta-trate-acm-SAGE-num-2000-0-20221228-232854-210836

CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model SAGE --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-SAGE-full-0-0-20221228-232308-116666/' \
--load_path './logs/Meta-trate-acm-SAGE-num-2000-0-20221228-232854-210836/'
12/28 11:36:11 PM Compare: LR target acc = [[0.69994646]] vs. Real target acc = 0.7158479690551758
12/28 11:36:12 PM Finish, this is the log dir = ./logs/metaLR-acm-to-dblp-SAGE-0-20221228-233608-725668
Test set: R2 :0.1084 RMSE: 0.1466 MAE: 0.1019

可能与Linear Regression拟合器相关



