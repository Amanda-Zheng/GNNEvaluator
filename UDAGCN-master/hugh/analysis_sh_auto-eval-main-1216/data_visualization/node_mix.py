from utils.GetFieldDataLists import get_field_data_lists
import matplotlib.pyplot as plt

dataLocationPrefix = "../data/node_mix"
best_source_acc_list, best_target_acc_list, dist_s_tra_aug_val_list, dist_aug_val_t_full_list, dist_s_val_t_full_list = get_field_data_lists(dataLocationPrefix)

plt.subplot(2, 1, 1)
plt.title("node_mixUp analysis")
plt.ylabel("distribution value")
plt.plot(dist_s_tra_aug_val_list, 'o:', color='r', label="dist_s_tra_aug_val")
plt.plot(dist_aug_val_t_full_list, 'o:', color='b', label="dist_aug_val_t_full")
plt.plot(dist_s_val_t_full_list, 'o:', color='g', label="dist_s_val_t_full")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.xlabel("sample id")
plt.ylabel("accuracy value")
plt.plot(best_source_acc_list, 'o:', color='c', label="best_source_acc")
plt.plot(best_target_acc_list, 'o:', color='m', label="best_target_acc")
plt.grid()
plt.legend()

plt.show()
