BEGIN {
    FS = "="
}

/dist_s_val_t_full/ {
    print($2)
}

END {
    # ------
    # The base source is abstract_info.txt
    # ------
}