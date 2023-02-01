import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../utils'))

from get_field_data import get_field_data  # type: ignore


def main():
    LOG_PATH = os.path.abspath('../../GAT/metaLR-acm-to-dblp-GAT-0-20230119-085903-270711/test.log')
    loss_test = get_field_data('loss_test', LOG_PATH)
    # R2 = get_field_data('R2', LOG_PATH)
    # RMSE = get_field_data('RMSE', LOG_PATH)
    # MAE = get_field_data('MAE', LOG_PATH)
    predictTarget = get_field_data('predict target', LOG_PATH)
    realTarget = get_field_data('real target', LOG_PATH)
    # assert len(loss_test) == len(R2) == len(RMSE) == len(MAE) == len(predictTarget) == len(realTarget)
    assert len(loss_test) == len(predictTarget) == len(realTarget)

    plt.subplot(2, 1, 1)
    plt.title("meta_regression_nn GAT", {'color': 'blue'})
    plt.ylabel("loss_test", {'color': 'darkred'})
    plt.plot(loss_test)

    plt.subplot(2, 1, 2)
    plt.ylabel('accuracy', {'color': 'darkred'})
    plt.xlabel('epoch number', {'color': 'green'})
    plt.plot(predictTarget, label="predict target")
    plt.plot(realTarget, label="real target")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
