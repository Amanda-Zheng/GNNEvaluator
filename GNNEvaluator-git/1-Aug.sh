#ACM-induct
CUDA_VISIBLE_DEVICES=1 python meta_set_save_induct-1.py --seed=0 --source acm --num_metas 400 --interval 100 200 300 400
##./logs/MetaSet/Meta-save-acm-num-400-0-20230427-092724-856325

#DBLP-induct
CUDA_VISIBLE_DEVICES=1 python meta_set_save_induct-1.py --seed=0 --source dblp --num_metas 400 --interval 100 200 300 400
##./logs/MetaSet/Meta-save-dblp-num-400-0-20230501-141442-956553

#Citation-network-induct
CUDA_VISIBLE_DEVICES=1 python meta_set_save_induct-1.py --seed=0 --source network --num_metas 400 --interval 100 200 300 400
##./logs/MetaSet/Meta-save-network-num-400-0-20230504-151757-268442





