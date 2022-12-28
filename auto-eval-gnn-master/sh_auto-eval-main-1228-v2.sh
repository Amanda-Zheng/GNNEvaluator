CUDA_VISIBLE_DEVICES=1 python pretrain-gcn.py --seed=0 \
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





