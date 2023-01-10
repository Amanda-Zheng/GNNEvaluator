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