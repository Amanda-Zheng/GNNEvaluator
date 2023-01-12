# Auto Evaluator for GNNs: A Graph Data-centric View

### Instructions
***
* step1: model pre-train some possible models

```
CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
--lr 5e-3 --wd 1e-5 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16
```

* step2: meta-set save to first create numbers of dataset and save it (data-centric without models)

```
CUDA_VISIBLE_DEVICES=1 python meta_set_save.py --seed=0 --source acm --num_metas 300
```
* step3: calculate meta-feat and acc for saving (model and data centric)

```
CUDA_VISIBLE_DEVICES=1 python meta_feat_acc.py --seed=0 --source acm \
--num_metas 300 --model GCN --hid_dim 128 --encoder_dim 16 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--aug_data_path './logs/Meta-save-acm-num-300-0-20221229-142336-945310/'
#./logs/Meta-feat-acc-acm-GCN-num-300-0-20221229-144816-935468
```
* step4: meta-regression to predict model on target dataset acc, decides model outputs

```
CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/' \
--load_path './logs/Meta-feat-acc-acm-GCN-num-300-0-20221229-144816-935468/'
#./logs/metaLR-acm-to-dblp-GCN-0-20221229-145546-042027
```