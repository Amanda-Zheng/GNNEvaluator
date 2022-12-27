echo "Starting to generate the abstraction data"
for dataType in $(ls -d node*);
do
  awk -f 'abstract_info.awk' $dataType/data.txt > $dataType/abstract_info.txt
  awk -f 'abstract_best_source_acc.awk' $dataType/abstract_info.txt > $dataType/abstract_best_source_acc.txt
  awk -f 'abstract_best_target_acc.awk' $dataType/abstract_info.txt > $dataType/abstract_best_target_acc.txt
  awk -f 'abstract_dist_s_tra_aug_val.awk' $dataType/abstract_info.txt > $dataType/abstract_dist_s_tra_aug_val.txt
  awk -f 'abstract_dist_aug_val_t_full.awk' $dataType/abstract_info.txt > $dataType/abstract_dist_aug_val_t_full.txt
  awk -f 'abstract_dist_s_val_t_full.awk' $dataType/abstract_info.txt > $dataType/abstract_dist_s_val_t_full.txt
done
echo "Finished :)"