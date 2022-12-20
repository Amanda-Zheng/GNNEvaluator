from .CreateListFromTextFile import create_list_from_text_file


def get_field_data_lists(dataLocationPrefix):
    best_source_acc_list = list(map(float, create_list_from_text_file("{}/abstract_best_source_acc.txt".format(dataLocationPrefix))))
    best_target_acc_list = list(map(float, create_list_from_text_file("{}/abstract_best_target_acc.txt".format(dataLocationPrefix))))
    dist_s_tra_aug_val_list = list(map(float, create_list_from_text_file("{}/abstract_dist_s_tra_aug_val.txt".format(dataLocationPrefix))))
    dist_aug_val_t_full_list = list(map(float, create_list_from_text_file("{}/abstract_dist_aug_val_t_full.txt".format(dataLocationPrefix))))
    dist_s_val_t_full_list = list(map(float, create_list_from_text_file("{}/abstract_dist_s_val_t_full.txt".format(dataLocationPrefix))))
    return best_source_acc_list, best_target_acc_list, dist_s_tra_aug_val_list, dist_aug_val_t_full_list, dist_s_val_t_full_list
