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
