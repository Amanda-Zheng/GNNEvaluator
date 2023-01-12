CUDA_VISIBLE_DEVICES=1 python meta_feat_acc.py --seed=0 --source acm \
--num_metas 300 --model GCN --hid_dim 128 --encoder_dim 16 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--aug_data_path './logs/Meta-save-acm-num-300-0-20221229-142336-945310/'
#./logs/Meta-feat-acc-acm-GCN-num-300-0-20230110-113023-190091

CUDA_VISIBLE_DEVICES=1 python meta_feat_acc.py --seed=0 --source acm \
--num_metas 300 --model SAGE --hid_dim 128 --encoder_dim 16 \
--model_path './logs/acm-to-dblp-SAGE-full-0-0-20221229-115452-301677/' \
--aug_data_path './logs/Meta-save-acm-num-300-0-20221229-142336-945310/'
#./logs/Meta-feat-acc-acm-SAGE-num-300-0-20230110-114021-468812

CUDA_VISIBLE_DEVICES=1 python meta_feat_acc.py --seed=0 --source acm \
--num_metas 10 --model GCN --hid_dim 128 --encoder_dim 16 --k_laplacian 2 --walk_length 0 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--aug_data_path './logs/Meta-save-acm-num-300-0-20221229-142336-945310/'
#./logs/Meta-feat-acc-acm-GCN-num-10-0-20230110-154700-625677

CUDA_VISIBLE_DEVICES=3 python meta_feat_acc.py --seed=0 --source acm \
--num_metas 300 --model GCN --hid_dim 128 --encoder_dim 16 --k_laplacian 5 --walk_length 0 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--aug_data_path './logs/Meta-save-acm-num-300-0-20221229-142336-945310/'
#./logs/Meta-feat-acc-acm-GCN-num-300-0-20230110-224213-828327


CUDA_VISIBLE_DEVICES=3 python meta_regression_nn.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 30 --reg_model 'mlp_simple' \
--k_laplacian 5 --epochs_reg 200 --lr_reg 0.1 --dropout_reg 0 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--load_path './logs/Meta-feat-acc-acm-GCN-num-300-0-20230110-224213-828327'
#./logs/metaLR-acm-to-dblp-GCN-0-20230112-161259-456626
#./logs/metaLR-acm-to-dblp-GCN-0-20230112-162005-870508

CUDA_VISIBLE_DEVICES=3 python meta_regression_nn.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 30 --reg_model 'mlp_simple' \
--k_laplacian 5 --epochs_reg 400 --lr_reg 0.1 --dropout_reg 0 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--load_path './logs/Meta-feat-acc-acm-GCN-num-300-0-20230110-224213-828327'
#./logs/metaLR-acm-to-dblp-GCN-0-20230112-161712-922519

CUDA_VISIBLE_DEVICES=3 python meta_regression_nn.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 30 --reg_model 'mlp_simple' \
--k_laplacian 5 --epochs_reg 400 --lr_reg 0.01 --dropout_reg 0 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--load_path './logs/Meta-feat-acc-acm-GCN-num-300-0-20230110-224213-828327'
#./logs/metaLR-acm-to-dblp-GCN-0-20230112-162151-134311

CUDA_VISIBLE_DEVICES=3 python meta_regression_nn.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 100 --reg_model 'mlp_simple' \
--k_laplacian 5 --epochs_reg 400 --lr_reg 0.01 --dropout_reg 0 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--load_path './logs/Meta-feat-acc-acm-GCN-num-300-0-20230110-224213-828327'
#./logs/metaLR-acm-to-dblp-GCN-0-20230112-162631-409065

CUDA_VISIBLE_DEVICES=3 python meta_regression_nn.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 10 --reg_model 'mlp_simple' \
--k_laplacian 5 --epochs_reg 400 --lr_reg 0.01 --dropout_reg 0 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--load_path './logs/Meta-feat-acc-acm-GCN-num-300-0-20230110-224213-828327'
#./logs/metaLR-acm-to-dblp-GCN-0-20230112-162915-108539