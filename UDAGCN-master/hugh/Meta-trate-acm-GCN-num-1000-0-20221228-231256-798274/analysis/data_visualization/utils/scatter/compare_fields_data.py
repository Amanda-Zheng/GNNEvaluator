import matplotlib.pyplot as plt


def compare_fields_data(x, y, done=False):
    plt.scatter(x, y)
    if done:
        return plt.show()
    else:
        return compare_fields_data


def _complete():
    plt.show()
