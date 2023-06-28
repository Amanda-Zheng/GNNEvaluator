# GNNEvaluator: Evaluating GNN Performance On Unseen Graphs Without Labels

This is the Pytorch implementation for "GNNEvaluator: Evaluating GNN Performance On Unseen Graphs Without Labels"

The framework is:



### Requirements
```
pyg==2.3.0 (py39_torch_1.13.0_cu117 )
pytorch==1.13.1 (py3.9_cuda11.7_cudnn8.5.0_0)
scikit-learn==1.2.2
torch-scatter==2.1.0+pt113cu117
torch-sparse==0.6.16+pt113cu117
```
## Instructions
### For evaluating your own well-trained GNNs 
(1) Run commands like in following to create simulated distribution shift from your source graphs (used for training your GNNs)
```
GNNEvaluator-git/1-Aug.sh
```
For instance,
```
python meta_set_save_induct-1.py --source=${yourdataset} --num_metas=${yournums} --interval=${yourinterval}
```
(2) Use the step-(1) obtained augmented graphs to create model-related Discgraph set with the unseen target graph with the commands like in:
```
GNNEvaluator-git/2-MetaG_Gen.sh
```
For instance,
```
python discrep_meta_graph-2.py --source=${yourdataset} --test_rate 0.2 \
--num_metas=${yournums} --interval=${yourinterval} \
--model=${your_evaluated_model} --hid_dim=${your_model_hid_dim} --encoder_dim=${your_model_en_dim} --num_layers=${your_model_layers} \
--model_path=${your_evaluated_model_path} \
--aug_data_path=${augmented-metaG-path-from-step(1)}
```
(3) Use the step-(2) obtained Discgraphs to train **GNNEvaluator** with the commands like in:
```
GNNEvaluator-git/3-GNN-evaluator.sh
```
For instance,
```
python meta_feat_acc_dist-3.py --num_metas=${yournums} --pre_epochs 50 \
--pre_lr=1e-7 --pre_drop=0.7 --early_stop_train 10 --early_stop 10 --pre_wd=0 --source=${yourdataset} --target=${your_tar_dataset} \
--train_batch_size=4 --test_batch_size=1  --seed=0 \
--model=${your_evaluated_model} --hid_dim=${your_model_hid_dim} --encoder_dim=${your_model_en_dim} --num_layers=${your_model_layers} --eval_out_dim 16 \
--model_path=${your_evaluated_model_path} \ 
--metaG_path=${DiscG-path-from-step(2)}  
```
### For reproducing our results

Following the above steps (1) to (3) with our hyper-parameters in file (contact Xin Zheng for access):
```
https://docs.google.com/spreadsheets/d/1SF-aY6usm9P0pZx0I888U3Xom0seH6BS3PU0vyqYTm0/edit?usp=sharing
```
If you would like to access our well-pretrained GNNs on ACM, DBLP, and Citation neworks, please contact: xin.zheng@monash.edu 

