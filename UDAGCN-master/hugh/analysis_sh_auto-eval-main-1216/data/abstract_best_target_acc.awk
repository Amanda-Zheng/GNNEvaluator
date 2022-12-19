BEGIN {
    FS = ": "
}

/best_target_acc/ {
    print($2)
}

END {
    # ------
    # The base source is abstract_info.txt
    # ------
}