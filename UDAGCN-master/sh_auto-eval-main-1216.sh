<<!
# 测试不同edge drop rate下的, meta-set 扰动的影响。
for p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  CUDA_VISIBLE_DEVICES=0 python auto-eval-main.py --seed=0 \
  --lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16 --edge_drop_all_p=${p}
done

## log_sh_auto-eval-main.txt
# 观察模型训练好的最佳结果的时候, meta-val与t-full之间的距离, s-train与meta-val之间的距离
# 但是这样缺乏一个baseline, s-val与t-full之间的距离
12/09 09:46:50 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 101, best_source_acc: 0.8097165822982788, best_target_acc: 0.6663678884506226, dist_s_tra_aug_val = 0.009999655187129974,dist_aug_val_t_full = 0.7655028104782104
12/09 09:47:26 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 101, best_source_acc: 0.8097165822982788, best_target_acc: 0.6663678884506226, dist_s_tra_aug_val = 0.020854484289884567,dist_aug_val_t_full = 0.694985032081604
12/09 09:48:04 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 155, best_source_acc: 0.8036437034606934, best_target_acc: 0.6665471196174622, dist_s_tra_aug_val = 0.03339676558971405,dist_aug_val_t_full = 0.7798579335212708
12/09 09:48:36 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 155, best_source_acc: 0.8036437034606934, best_target_acc: 0.6665471196174622, dist_s_tra_aug_val = 0.04743853211402893,dist_aug_val_t_full = 0.7949701547622681
12/09 09:49:11 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 101, best_source_acc: 0.8097165822982788, best_target_acc: 0.6663678884506226, dist_s_tra_aug_val = 0.07830439507961273,dist_aug_val_t_full = 0.6284030675888062
12/09 09:49:50 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 155, best_source_acc: 0.8036437034606934, best_target_acc: 0.6665471196174622, dist_s_tra_aug_val = 0.08017702400684357,dist_aug_val_t_full = 0.7506200075149536
12/09 09:50:20 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 155, best_source_acc: 0.8036437034606934, best_target_acc: 0.6665471196174622, dist_s_tra_aug_val = 0.09133754670619965,dist_aug_val_t_full = 0.7615581154823303

## log_sh_auto-eval-main.txt
12/09 09:59:32 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 155, best_source_acc: 0.8036437034606934, best_target_acc: 0.6665471196174622, dist_s_tra_aug_val = 0.008888784795999527,dist_aug_val_t_full = 0.9586369395256042, dist_s_val_t_full=1.0565491914749146
12/09 10:00:17 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 101, best_source_acc: 0.8097165822982788, best_target_acc: 0.6663678884506226, dist_s_tra_aug_val = 0.020854484289884567,dist_aug_val_t_full = 0.694985032081604, dist_s_val_t_full=0.8517730236053467
12/09 10:00:52 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 155, best_source_acc: 0.8036437034606934, best_target_acc: 0.6665471196174622, dist_s_tra_aug_val = 0.03339564800262451,dist_aug_val_t_full = 0.7798671722412109, dist_s_val_t_full=1.0565568208694458
12/09 10:01:22 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 101, best_source_acc: 0.8097165822982788, best_target_acc: 0.6663678884506226, dist_s_tra_aug_val = 0.05543919652700424,dist_aug_val_t_full = 0.6449740529060364, dist_s_val_t_full=0.8517730236053467
12/09 10:01:56 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 101, best_source_acc: 0.8097165822982788, best_target_acc: 0.6663678884506226, dist_s_tra_aug_val = 0.07830438017845154,dist_aug_val_t_full = 0.6284030675888062, dist_s_val_t_full=0.8517730236053467
12/09 10:02:28 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 101, best_source_acc: 0.8097165822982788, best_target_acc: 0.6663678884506226, dist_s_tra_aug_val = 0.09239710867404938,dist_aug_val_t_full = 0.6098291873931885, dist_s_val_t_full=0.8517730236053467
12/09 10:03:02 PM source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 155, best_source_acc: 0.8036437034606934, best_target_acc: 0.6665471196174622, dist_s_tra_aug_val = 0.09133753180503845,dist_aug_val_t_full = 0.7615582346916199, dist_s_val_t_full=1.0565494298934937
!

for lamb in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  CUDA_VISIBLE_DEVICES=2 python auto-eval-main.py --seed=0 \
  --lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16 --aug_method 'node_mix' \
  --mix_lamb=${lamb}
done
for p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  CUDA_VISIBLE_DEVICES=2 python auto-eval-main.py --seed=0 \
  --lr 5e-3 --epochs 300 --model GCN --full_s 0 --encoder_dim 16 --aug_method 'node_fmask' \
  --node_fmask_all_p=${p}
done
2022-12-17 00:09:09,078 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 44, best_source_acc: 0.8238866329193115, best_target_acc: 0.6964861750602722, dist_s_tra_aug_val = 0.48190054297447205,dist_aug_val_t_full = 0.3955017924308777, dist_s_val_t_full=0.7430351972579956
2022-12-17 00:09:09,078 Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.1, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.05, seed=0, source='acm', target='dblp')
2022-12-17 00:09:09,078 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_mix-20221217-000855-182386
2022-12-17 00:09:26,190 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 44, best_source_acc: 0.8238866329193115, best_target_acc: 0.6964861750602722, dist_s_tra_aug_val = 0.49908432364463806,dist_aug_val_t_full = 0.3920787572860718, dist_s_val_t_full=0.7430351972579956
2022-12-17 00:09:26,190 Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.2, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.05, seed=0, source='acm', target='dblp')
2022-12-17 00:09:26,190 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_mix-20221217-000912-396253
2022-12-17 00:09:43,666 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 44, best_source_acc: 0.8238866329193115, best_target_acc: 0.6964861750602722, dist_s_tra_aug_val = 0.46536731719970703,dist_aug_val_t_full = 0.370816171169281, dist_s_val_t_full=0.7430351972579956
2022-12-17 00:09:43,666 Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.3, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.05, seed=0, source='acm', target='dblp')
2022-12-17 00:09:43,666 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_mix-20221217-000930-057091
2022-12-17 00:10:01,444 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 44, best_source_acc: 0.8238866329193115, best_target_acc: 0.6964861750602722, dist_s_tra_aug_val = 0.3877149820327759,dist_aug_val_t_full = 0.3508865535259247, dist_s_val_t_full=0.7430355548858643
2022-12-17 00:10:01,444 Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.4, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.05, seed=0, source='acm', target='dblp')
2022-12-17 00:10:01,445 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_mix-20221217-000947-354194
2022-12-17 00:10:19,197 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 44, best_source_acc: 0.8238866329193115, best_target_acc: 0.6964861750602722, dist_s_tra_aug_val = 0.28708934783935547,dist_aug_val_t_full = 0.35548773407936096, dist_s_val_t_full=0.7430351972579956
2022-12-17 00:10:19,197 Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.5, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.05, seed=0, source='acm', target='dblp')
2022-12-17 00:10:19,197 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_mix-20221217-001004-839412
2022-12-17 00:10:36,478 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 44, best_source_acc: 0.8238866329193115, best_target_acc: 0.6964861750602722, dist_s_tra_aug_val = 0.1871778964996338,dist_aug_val_t_full = 0.39626777172088623, dist_s_val_t_full=0.7430351972579956
2022-12-17 00:10:36,478 Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.6, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.05, seed=0, source='acm', target='dblp')
2022-12-17 00:10:36,478 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_mix-20221217-001022-104587
2022-12-17 00:10:56,468 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 44, best_source_acc: 0.8238866329193115, best_target_acc: 0.6964861750602722, dist_s_tra_aug_val = 0.10535138845443726,dist_aug_val_t_full = 0.46937716007232666, dist_s_val_t_full=0.7430351972579956
2022-12-17 00:10:56,468 Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.7, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.05, seed=0, source='acm', target='dblp')
2022-12-17 00:10:56,468 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_mix-20221217-001039-943874
2022-12-17 00:11:20,901 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 44, best_source_acc: 0.8238866329193115, best_target_acc: 0.6964861750602722, dist_s_tra_aug_val = 0.04893341660499573,dist_aug_val_t_full = 0.5617672204971313, dist_s_val_t_full=0.7430351972579956
2022-12-17 00:11:20,902 Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.8, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.05, seed=0, source='acm', target='dblp')
2022-12-17 00:11:20,902 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_mix-20221217-001102-117275
2022-12-17 00:11:38,960 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 44, best_source_acc: 0.8238866329193115, best_target_acc: 0.6964861750602722, dist_s_tra_aug_val = 0.016963660717010498,dist_aug_val_t_full = 0.6604522466659546, dist_s_val_t_full=0.7430355548858643
2022-12-17 00:11:38,960 Namespace(UDAGCN=False, aug_method='node_mix', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.9, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.05, seed=0, source='acm', target='dblp')
2022-12-17 00:11:38,960 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_mix-20221217-001126-958413
========
2022-12-17 00:11:51,275 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 99, best_source_acc: 0.8157894611358643, best_target_acc: 0.7068842053413391, dist_s_tra_aug_val = 0.029066674411296844,dist_aug_val_t_full = 0.6761600375175476, dist_s_val_t_full=0.8541948795318604
2022-12-17 00:11:51,275 Namespace(UDAGCN=False, aug_method='node_fmask', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.3, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.1, seed=0, source='acm', target='dblp')
2022-12-17 00:11:51,275 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_fmask-20221217-001141-612322
2022-12-17 00:12:06,633 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 99, best_source_acc: 0.8157894611358643, best_target_acc: 0.7068842053413391, dist_s_tra_aug_val = 0.10880924761295319,dist_aug_val_t_full = 0.47274279594421387, dist_s_val_t_full=0.8541958928108215
2022-12-17 00:12:06,634 Namespace(UDAGCN=False, aug_method='node_fmask', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.3, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.2, seed=0, source='acm', target='dblp')
2022-12-17 00:12:06,634 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_fmask-20221217-001153-815222
2022-12-17 00:12:23,544 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 99, best_source_acc: 0.8157894611358643, best_target_acc: 0.7068842053413391, dist_s_tra_aug_val = 0.1989704966545105,dist_aug_val_t_full = 0.3538873493671417, dist_s_val_t_full=0.8541960716247559
2022-12-17 00:12:23,544 Namespace(UDAGCN=False, aug_method='node_fmask', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.3, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.3, seed=0, source='acm', target='dblp')
2022-12-17 00:12:23,544 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_fmask-20221217-001210-690950
2022-12-17 00:12:40,847 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 99, best_source_acc: 0.8157894611358643, best_target_acc: 0.7068842053413391, dist_s_tra_aug_val = 0.35386011004447937,dist_aug_val_t_full = 0.2209586203098297, dist_s_val_t_full=0.8541958928108215
2022-12-17 00:12:40,847 Namespace(UDAGCN=False, aug_method='node_fmask', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.3, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.4, seed=0, source='acm', target='dblp')
2022-12-17 00:12:40,847 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_fmask-20221217-001227-455786
2022-12-17 00:12:57,295 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 99, best_source_acc: 0.8157894611358643, best_target_acc: 0.7068842053413391, dist_s_tra_aug_val = 0.5686835050582886,dist_aug_val_t_full = 0.1308692991733551, dist_s_val_t_full=0.8541958332061768
2022-12-17 00:12:57,295 Namespace(UDAGCN=False, aug_method='node_fmask', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.3, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.5, seed=0, source='acm', target='dblp')
2022-12-17 00:12:57,295 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_fmask-20221217-001244-278933
2022-12-17 00:13:14,192 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 99, best_source_acc: 0.8157894611358643, best_target_acc: 0.7068842053413391, dist_s_tra_aug_val = 0.7771122455596924,dist_aug_val_t_full = 0.12912414968013763, dist_s_val_t_full=0.8541949391365051
2022-12-17 00:13:14,192 Namespace(UDAGCN=False, aug_method='node_fmask', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.3, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.6, seed=0, source='acm', target='dblp')
2022-12-17 00:13:14,192 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_fmask-20221217-001300-849555
2022-12-17 00:13:30,364 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 99, best_source_acc: 0.8157894611358643, best_target_acc: 0.7068842053413391, dist_s_tra_aug_val = 1.0782313346862793,dist_aug_val_t_full = 0.2481905221939087, dist_s_val_t_full=0.8541953563690186
2022-12-17 00:13:30,364 Namespace(UDAGCN=False, aug_method='node_fmask', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.3, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.7, seed=0, source='acm', target='dblp')
2022-12-17 00:13:30,364 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_fmask-20221217-001317-288234
2022-12-17 00:13:46,372 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 99, best_source_acc: 0.8157894611358643, best_target_acc: 0.7068842053413391, dist_s_tra_aug_val = 1.3558847904205322,dist_aug_val_t_full = 0.464755117893219, dist_s_val_t_full=0.8541948199272156
2022-12-17 00:13:46,372 Namespace(UDAGCN=False, aug_method='node_fmask', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.3, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.8, seed=0, source='acm', target='dblp')
2022-12-17 00:13:46,372 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_fmask-20221217-001333-958297
2022-12-17 00:14:02,022 source: acm, target: dblp, seed: 0, UDAGCN: False, encoder_dim: 16 - Epoch: 99, best_source_acc: 0.8157894611358643, best_target_acc: 0.7068842053413391, dist_s_tra_aug_val = 1.6147100925445557,dist_aug_val_t_full = 0.7629371285438538, dist_s_val_t_full=0.8541938066482544
2022-12-17 00:14:02,023 Namespace(UDAGCN=False, aug_method='node_fmask', edge_drop_all_p=0.05, encoder_dim=16, epochs=300, full_s=0, lr=0.005, mix_lamb=0.3, model='GCN', name='UDAGCN', node_drop_val_p=0.05, node_fmask_all_p=0.9, seed=0, source='acm', target='dblp')
2022-12-17 00:14:02,023 Finish!, this is the log dir: ./logs/acm-to-dblp-GCN-full-0-0-aug-node_fmask-20221217-001349-611045
