BEGIN {
    # ------
    # Generated by running the command:
    # `awk -f abstract_dist_aug_val_t_full.awk ./[folder]/abstract_info.txt > ./[folder]/abstract_dist_aug_val_t_full.txt`
    # ------
    FS = " = "
}

/dist_aug_val_t_full/ {
    print($2)
}

END {
    # ------
    # The base source is abstract_info.txt
    # ------
}