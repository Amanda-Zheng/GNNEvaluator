#****************************************Step-1-MetaG-Generation*******************************************
for path in './logs/Models_tra/dblp-to-network-GCN-full-0-0-20230504-134557-731013' \
'./logs/Models_tra/dblp-to-network-GCN-full-0-1-20230504-134606-179266' \
'./logs/Models_tra/dblp-to-network-GCN-full-0-2-20230504-134614-894260' \
'./logs/Models_tra/dblp-to-network-GCN-full-0-3-20230504-134623-495759' \
'./logs/Models_tra/dblp-to-network-GCN-full-0-4-20230504-134632-190093'
do
CUDA_VISIBLE_DEVICES=1 python discrep_meta_graph-2.py --seed=0 --source dblp --test_rate 0.2 \
--num_metas 400 --interval 100 200 300 400 \
--model GCN --hid_dim 128 --encoder_dim 16 --num_layers 2 \
--model_path=${path} \
--aug_data_path='./logs/MetaSet/Meta-save-dblp-num-400-0-20230501-141442-956553/'
done
for path in './logs/Models_tra/dblp-to-network-SAGE-full-0-0-20230504-134641-121804' \
'./logs/Models_tra/dblp-to-network-SAGE-full-0-1-20230504-134655-693672' \
'./logs/Models_tra/dblp-to-network-SAGE-full-0-2-20230504-134709-899932' \
'./logs/Models_tra/dblp-to-network-SAGE-full-0-3-20230504-134724-053062' \
'./logs/Models_tra/dblp-to-network-SAGE-full-0-4-20230504-134738-437393'
do
CUDA_VISIBLE_DEVICES=1 python discrep_meta_graph-2.py --seed=0 --source dblp --test_rate 0.2 \
--num_metas 400 --interval 100 200 300 400 \
--model SAGE --hid_dim 128 --encoder_dim 16 --num_layers 2 \
--model_path=${path} \
--aug_data_path='./logs/MetaSet/Meta-save-dblp-num-400-0-20230501-141442-956553/' 
done
for path in './logs/Models_tra/dblp-to-network-GAT-full-0-0-20230504-134752-896420' \
'./logs/Models_tra/dblp-to-network-GAT-full-0-1-20230504-134802-265497' \
'./logs/Models_tra/dblp-to-network-GAT-full-0-2-20230504-134811-376104' \
'./logs/Models_tra/dblp-to-network-GAT-full-0-3-20230504-134820-744845' \
'./logs/Models_tra/dblp-to-network-GAT-full-0-4-20230504-134829-893638'
do
CUDA_VISIBLE_DEVICES=1 python discrep_meta_graph-2.py --seed=0 --source dblp --test_rate 0.2 \
--num_metas 400 --interval 100 200 300 400 \
--model GAT --hid_dim 128 --encoder_dim 16 --num_layers 2 \
--model_path=${path} \
--aug_data_path='./logs/MetaSet/Meta-save-dblp-num-400-0-20230501-141442-956553/' 
done
for path in './logs/Models_tra/dblp-to-network-GIN-full-0-0-20230504-134839-233147' \
'./logs/Models_tra/dblp-to-network-GIN-full-0-1-20230504-134853-692358' \
'./logs/Models_tra/dblp-to-network-GIN-full-0-2-20230504-134907-915868' \
'./logs/Models_tra/dblp-to-network-GIN-full-0-3-20230504-134922-464714' \
'./logs/Models_tra/dblp-to-network-GIN-full-0-4-20230504-134936-605209'
do
CUDA_VISIBLE_DEVICES=1 python discrep_meta_graph-2.py --seed=0 --source dblp --test_rate 0.2 \
--num_metas 400 --interval 100 200 300 400 \
--model GIN --hid_dim 128 --encoder_dim 16 --num_layers 2 \
--model_path=${path} \
--aug_data_path='./logs/MetaSet/Meta-save-dblp-num-400-0-20230501-141442-956553/' 
done
for path in './logs/Models_tra/dblp-to-network-MLP-full-0-0-20230504-134950-655092' \
'./logs/Models_tra/dblp-to-network-MLP-full-0-1-20230504-134957-510806' \
'./logs/Models_tra/dblp-to-network-MLP-full-0-2-20230504-135004-122932' \
'./logs/Models_tra/dblp-to-network-MLP-full-0-3-20230504-135011-091792' \
'./logs/Models_tra/dblp-to-network-MLP-full-0-4-20230504-135018-286769'
do
CUDA_VISIBLE_DEVICES=1 python discrep_meta_graph-2.py --seed=0 --source dblp --test_rate 0.2 \
--num_metas 400 --interval 100 200 300 400 \
--model MLP --hid_dim 128 --encoder_dim 16 --num_layers 2 \
--model_path=${path} \
--aug_data_path='./logs/MetaSet/Meta-save-dblp-num-400-0-20230501-141442-956553/' 
done
