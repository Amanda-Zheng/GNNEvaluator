# step 1
CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 5e-3 --wd 1e-5 --epochs 200 --model GIN --full_s 0 --hid_dim 128 --encoder_dim 16
# --> get model_path
#2023-01-19 09:08:52,009 Best Epoch: 56, best_source_test_acc: 0.7672064304351807, best_source_val_acc: 0.8151146769523621, best_target_acc: 0.5629257559776306
#2023-01-19 09:08:52,009 Namespace(encoder_dim=16, epochs=200, full_s=0, hid_dim=128, lr=0.005, model='GIN', seed=0, source='acm', target='dblp', wd=1e-05)
#2023-01-19 09:08:52,010 Finish!, this is the log dir = ./logs/acm-to-dblp-GIN-full-0-0-20230119-090839-975822

:"
# step 2
CUDA_VISIBLE_DEVICES=1 python meta_set_save.py --seed=0 --source acm --num_metas 300
# --> get aug_data_path
#./logs/Meta-save-acm-num-300-0-20230117-162906-177221
"

:"
# step 3
CUDA_VISIBLE_DEVICES=1 python meta_feat_acc.py --seed=0 --source acm \
--num_metas 300 --model GIN --hid_dim 128 --encoder_dim 16 \
--model_path '$??' \
--aug_data_path './logs/Meta-save-acm-num-300-0-20230117-162906-177221' # use meta-set we have created
# --> get load_path

# step 4
CUDA_VISIBLE_DEVICES=1 python meta_regression_nn.py --seed=0 \
	--source acm --target dblp --model GIN --hid_dim 128 --encoder_dim 16 --val_num 30 --reg_model 'mlp_simple'\
	--k_laplacian 5 --epochs_reg 200 --lr_reg 0.1 --dropout_reg 0 \
	--model_path '$??' \
	--load_path '$??'
# --> get our 'super model' which is trained based on the original given model

# step 4
CUDA_VISIBLE_DEVICES=1 python meta_regression_nn.py --seed=0 \
	--source acm --target dblp --model GIN --hid_dim 128 --encoder_dim 16 --val_num 30 --reg_model 'mlp_simple'\
	--k_laplacian 5 --epochs_reg 400 --lr_reg 0.1 --dropout_reg 0 \
	--model_path '$??' \
	--load_path '$??'
# --> get our 'super model' which is trained based on the original given model
"
