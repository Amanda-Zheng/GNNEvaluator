:"
# step 1
CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 5e-3 --wd 1e-5 --epochs 200 --model GAT --full_s 0 --hid_dim 128 --encoder_dim 16
# --> get model_path
#2023-01-17 15:59:11,618 Best Epoch: 27, best_source_test_acc: 0.8319838047027588, best_source_val_acc: 0.8353576064109802, best_target_acc: 0.5516313910484314
#2023-01-17 15:59:11,618 Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.005, model='GAT', seed=0, source='acm', target='dblp', wd=1e-05)
#2023-01-17 15:59:11,619 Finish!, this is the log dir = ./logs/acm-to-dblp-GAT-full-0-0-20230117-155857-034722
"

# step 2
CUDA_VISIBLE_DEVICES=1 python meta_set_save.py --seed=0 --source acm --num_metas 300
# --> get aug_data_path
#./logs/Meta-save-acm-num-300-0-20230117-162906-177221

:"
# step 3
CUDA_VISIBLE_DEVICES=1 python meta_feat_acc.py --seed=0 --source acm \
--num_metas 300 --model GAT --hid_dim 128 --encoder_dim 16 \
--model_path $model_path \
--aug_data_path $aug_data_path
# --> get load_path

# step 4
CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model GAT --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path $model_path \
--load_path $load_path
# --> get our 'super model' which is trained based on the original given model
"