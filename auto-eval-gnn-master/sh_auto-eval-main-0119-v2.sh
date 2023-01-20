:"
# step 1
CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
  --lr 5e-3 --wd 1e-5 --epochs 200 --model MLP --full_s 0 --hid_dim 128 --encoder_dim 16
# --> get model_path
#./logs/acm-to-dblp-MLP-full-0-0-20230119-222614-746470

# step 2
CUDA_VISIBLE_DEVICES=1 python meta_set_save.py --seed=0 --source acm --num_metas 300
# --> get aug_data_path
#./logs/Meta-save-acm-num-300-0-20230117-162906-177221

# step 3
CUDA_VISIBLE_DEVICES=1 python meta_feat_acc.py --seed=0 --source acm \
--num_metas 300 --model MLP --hid_dim 128 --encoder_dim 16 \
--model_path './logs/acm-to-dblp-MLP-full-0-0-20230119-222614-746470' \
--aug_data_path './logs/Meta-save-acm-num-300-0-20230117-162906-177221' # use meta-set we have created
# --> get load_path
#./logs/Meta-feat-acc-acm-MLP-num-300-0-20230120-151727-919480

# step 4
CUDA_VISIBLE_DEVICES=1 python meta_regression_nn.py --seed=0 \
	--source acm --target dblp --model MLP --hid_dim 128 --encoder_dim 16 --val_num 30 --reg_model 'mlp_simple'\
	--k_laplacian 5 --epochs_reg 200 --lr_reg 0.1 --dropout_reg 0 \
	--model_path './logs/acm-to-dblp-MLP-full-0-0-20230119-222614-746470' \
	--load_path './logs/Meta-feat-acc-acm-MLP-num-300-0-20230120-151727-919480'
# --> get our 'super model' which is trained based on the original given model
#./logs/metaLR-acm-to-dblp-MLP-0-20230120-224611-499525

# step 4
CUDA_VISIBLE_DEVICES=1 python meta_regression_nn.py --seed=0 \
	--source acm --target dblp --model MLP --hid_dim 128 --encoder_dim 16 --val_num 30 --reg_model 'mlp_simple'\
	--k_laplacian 5 --epochs_reg 400 --lr_reg 0.1 --dropout_reg 0 \
	--model_path './logs/acm-to-dblp-MLP-full-0-0-20230119-222614-746470' \
	--load_path './logs/Meta-feat-acc-acm-MLP-num-300-0-20230120-151727-919480'
# --> get our 'super model' which is trained based on the original given model
#./logs/metaLR-acm-to-dblp-MLP-0-20230120-224754-405418
"
