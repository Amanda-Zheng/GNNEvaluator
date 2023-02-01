import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../utils'))

from get_field_data import get_field_data  # type: ignore


def main():
    PRE_TRAIN_TEST_LOG = os.path.abspath('../../GAT/Meta-feat-acc-acm-GAT-num-300-0-20230117-175809-721176/test.log')
    FEAT = get_field_data('FEAT', PRE_TRAIN_TEST_LOG)
    ACC = get_field_data('ACC', PRE_TRAIN_TEST_LOG)
    groups = group_data(FEAT, ACC)  # -> [(FEAT, ACC), ...]
    sortedGroups = sorted(groups, key=sorted_aux, reverse=True)
    sortedFEAT, sortedACC = destruct_group_data(sortedGroups)

    plt.subplot(2, 1, 1)
    plt.title("meta_feat_acc GAT", {'color': 'blue'})
    plt.ylabel("FEAT", {'color': 'darkred'})
    plt.plot(sortedFEAT)

    plt.subplot(2, 1, 2)
    plt.ylabel('ACC', {'color': 'darkred'})
    plt.xlabel('samples', {'color': 'green'})
    plt.plot(sortedACC)

    plt.show()


def sorted_aux(group):
    """
    Args:
        group: (A, B)
    Returns: A
    """
    return group[0]


def group_data(listA, listB):
    assert len(listA) == len(listB)

    return [(listA[i], listB[i]) for i in range(len(listA))]


def destruct_group_data(listGroup):
    """
    Args:
        listGroup: [(A,B), ...]
    Returns: [A,...], [B,...]
    """
    listA = [listGroup[i][0] for i in range(len(listGroup))]
    listB = [listGroup[i][1] for i in range(len(listGroup))]

    return listA, listB


if __name__ == "__main__":
    main()