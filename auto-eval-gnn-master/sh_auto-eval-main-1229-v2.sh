for lr in 1e-2 3e-3 5e-3
do
  for wd in 1e-5 1e-4 5e-4 1e-3 5e-3
  do
  CUDA_VISIBLE_DEVICES=1 python pretrain_gnn.py --seed=0 --source acm --target dblp \
  --lr=${lr} --wd=${wd} --epochs 300 --model SAGE --full_s 0 --hid_dim 128 --encoder_dim 16
  done
done

CUDA_VISIBLE_DEVICES=1 python meta_train_test.py --seed=0 \
--num_metas 300 --model SAGE --hid_dim 128 --encoder_dim 16 \
--save_path './logs/acm-to-dblp-SAGE-full-0-0-20221229-115452-301677'
12/29 01:55:36 PM Namespace(aug_method='node_mix', edge_drop_all_p=0.8532459761585128, encoder_dim=16, hid_dim=128, mix_lamb=0.9692058717176852, model='SAGE', node_drop_val_p=0.05, node_fmask_all_p=0.4753247822120824, num_metas=300, save_path='./logs/acm-to-dblp-SAGE-full-0-0-20221229-115452-301677', seed=0, source='acm')
12/29 01:55:36 PM Finish, this is the log dir = ./logs/Meta-trate-acm-SAGE-num-300-0-20221229-135512-960793

CUDA_VISIBLE_DEVICES=1 python meta_regression.py --seed=0 \
--source acm --target dblp --model SAGE --hid_dim 128 --encoder_dim 16 --val_num 30 \
--model_path './logs/acm-to-dblp-SAGE-full-0-0-20221229-115452-301677/' \
--load_path './logs/Meta-trate-acm-SAGE-num-300-0-20221229-135512-960793/'