BEGIN {
    FS=",";
    FS = "(, |,)"
}

/best_source_acc/ {
    print("record line: " NR)
    start = 7
    fieldCounts = NF
    do {
        print($start)
        start += 1
    } while (start <= fieldCounts)
    print("\n")
}

END {
    # "------"
    # "The base source is data.txt"
    # "------"
}