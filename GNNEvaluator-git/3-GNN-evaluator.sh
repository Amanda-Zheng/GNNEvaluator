#!/bin/bash
array1=(
'./logs/Models_tra/dblp-to-network-GCN-full-0-0-20230504-134557-731013'
'./logs/Models_tra/dblp-to-network-GCN-full-0-1-20230504-134606-179266'
'./logs/Models_tra/dblp-to-network-GCN-full-0-2-20230504-134614-894260'
'./logs/Models_tra/dblp-to-network-GCN-full-0-3-20230504-134623-495759'
'./logs/Models_tra/dblp-to-network-GCN-full-0-4-20230504-134632-190093'
)
array2=(
'./logs/MetaG/Meta-feat-acc-dblp-GCN-num-400-0-20230504-225535-427535'
'./logs/MetaG/Meta-feat-acc-dblp-GCN-num-400-0-20230504-230659-413751'
'./logs/MetaG/Meta-feat-acc-dblp-GCN-num-400-0-20230504-231522-103289'
'./logs/MetaG/Meta-feat-acc-dblp-GCN-num-400-0-20230504-232236-733584'
'./logs/MetaG/Meta-feat-acc-dblp-GCN-num-400-0-20230504-232858-852941'
)
for i in "${!array1[@]}"; do
  CUDA_VISIBLE_DEVICES=3 python meta_feat_acc_dist-3.py --num_metas=400 --pre_epochs 50 \
  --pre_lr=1e-7 --pre_drop=0.7 --early_stop_train 10 --early_stop 10 --pre_wd=0 --source dblp --target acm --target2 network \
  --train_batch_size=4 --test_batch_size=1  --seed=0 \
  --model GCN --hid_dim 128 --encoder_dim 16 --eval_hid_dim 128 --eval_out_dim 16 --num_layers 2 \
  --model_path=${array1[$i]} \
  --metaG_path=${array2[$i]}
done
