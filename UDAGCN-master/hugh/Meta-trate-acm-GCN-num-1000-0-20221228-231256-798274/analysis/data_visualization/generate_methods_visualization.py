import os

import matplotlib.pyplot as plt

from utils.get_field_data import get_field_data


def generate_methods_visualization(folderLocation):
    methodFileLocations = os.listdir(folderLocation)
    FEAT_FIELD = 'FEAT'
    ACC_FIELD = 'ACC'
    for file in methodFileLocations:
        feats = get_field_data(FEAT_FIELD, "{}/{}".format(folderLocation, file))
        accs = get_field_data(ACC_FIELD, "{}/{}".format(folderLocation, file))
        plt.annotate(file, xy=(feats[0], accs[0]), xytext=(feats[0] + 0.1, accs[0] + 0.05),
                     arrowprops=dict(facecolor='black', headwidth=6, width=0.4, shrink=0.05))
        plt.scatter(feats, accs)

    plt.title("meta_acc VS meta_feat", color='r')
    plt.xlabel("meta_feat")
    plt.ylabel("meta_acc")
    plt.show()


if __name__ == '__main__':
    generate_methods_visualization('../data/methods/sorted')
