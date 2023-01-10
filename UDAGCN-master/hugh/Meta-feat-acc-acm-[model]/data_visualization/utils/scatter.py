import matplotlib.pyplot as plt


def compare_fields_data(xLabel, x, yLabel, y, done=False):
    plt.scatter(x, y, label="{}:{}".format(yLabel, xLabel), alpha=1, edgecolors='none')

    if done:
        return _complete()
    else:
        return compare_fields_data


def _complete():
    return
