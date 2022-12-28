# Auto Evaluator for GNNs: A Graph Data-centric View

### Instructions
***
* step1: model pre-train

```
CUDA_VISIBLE_DEVICES=1 python pretrain-gcn.py --seed=0 \
--lr 5e-3 --wd 5e-4 --epochs 200 --model GCN --full_s 0 --hid_dim 128 --encoder_dim 16 
```

* step2: meta-train-test to first create numbers of dataset, \
and then calculate mmd metric as feat and model's prediction as acc \
needs loading pre-trained models


```
CUDA_VISIBLE_DEVICES=1 python meta_train_test.py --seed=0 \
--num_metas 300 --model GCN --hid_dim 128 --encoder_dim 16 \
--save_path './logs/acm-GCN-full-0-0-20221228-213830-683591/'
```

* step3: meta-regression to predict model on target dataset acc, decides model outputs

```
CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model GCN --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-GCN-full-0-0-20221228-213830-683591/' \
--load_path './logs/Meta-trate-acm-GCN-num-300-0-20221228-230057-132889/'
```