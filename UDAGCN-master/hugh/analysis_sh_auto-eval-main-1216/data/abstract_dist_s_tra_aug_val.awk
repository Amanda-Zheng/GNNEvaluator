BEGIN {
    FS = " = "
}

/dist_s_tra_aug_val/ {
    print($2)
}

END {
    # ------
    # The base source is abstract_info.txt
    # ------
}