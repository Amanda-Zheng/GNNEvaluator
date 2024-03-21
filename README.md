# GNNEvaluator: Evaluating GNN Performance On Unseen Graphs Without Labels

This is the Pytorch implementation for  NeurIPS-23:"GNNEvaluator: Evaluating GNN Performance On Unseen Graphs Without Labels"
We are trying to solve the GNN evaluation problem when serving on unseen graphs without labels as:

![pre4](https://github.com/Amanda-Zheng/GNNEvaluator/assets/61812981/7e79b25d-429f-445e-9a3f-781e9d703234)

The framework is:

![GNN_eval9](https://github.com/Amanda-Zheng/GNNEvaluator/assets/61812981/c7f5f661-274e-4b01-951a-4417dc75a802)

Welcome to kindly cite our work and discuss with xin.zheng@monash.edu:

```
@article{zheng2023gnnevaluator,
  title={GNNEvaluator: Evaluating GNN Performance On Unseen Graphs Without Labels},
  author={Zheng, Xin and Zhang, Miao and Chen, Chunyang and Molaei, Soheila and Zhou, Chuan and Pan, Shirui},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

### Requirements
```
pyg==2.3.0 (py39_torch_1.13.0_cu117 )
pytorch==1.13.1 (py3.9_cuda11.7_cudnn8.5.0_0)
scikit-learn==1.2.2
torch-scatter==2.1.0+pt113cu117
torch-sparse==0.6.16+pt113cu117
```
## Instructions
For the dataset, please check: https://drive.google.com/drive/folders/1RqfrAaXdINmklxbByKUoJxWCD1yfs_c_?usp=sharing
### For evaluating your own well-trained GNNs 
(1) Run commands like in following to create a simulated distribution shift from your source graphs (used for training your GNNs)
```
GNNEvaluator-git/1-Aug.sh
```
For instance,
```
python meta_set_save_induct-1.py --source=${yourdataset} --num_metas=${yournums} --interval=${yourinterval}
```
(2) Use step-(1) obtained augmented graphs to create a model-related Discgraph set with the unseen target graph with the commands like in:
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

