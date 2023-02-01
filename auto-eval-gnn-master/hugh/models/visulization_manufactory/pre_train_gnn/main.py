import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../utils'))

from get_field_data import get_field_data  # type: ignore

modelName = 'GAT'
modelLogPath = '../../GAT/acm-to-dblp-GAT-full-0-0-20230117-155857-034722/test.log'


def main():
    LOG_PATH = os.path.abspath(modelLogPath)
    source_train_loss = get_field_data('source_train_loss', LOG_PATH)
    source_train_acc = get_field_data('source_train_acc', LOG_PATH)
    source_val_acc = get_field_data('source_val_acc', LOG_PATH)
    source_test_acc = get_field_data('source_test_acc', LOG_PATH)
    assert len(source_train_loss) == len(source_train_acc) == len(source_val_acc) == len(source_test_acc)

    plt.subplot(2, 1, 1)
    plt.title("pre_train {}".format(modelName), {'color': 'blue'})
    plt.ylabel("source_train_loss", {'color': 'darkred'})
    plt.plot(source_train_loss)

    plt.subplot(2, 1, 2)
    plt.ylabel('accuracy', {'color': 'darkred'})
    plt.xlabel('epoch number', {'color': 'green'})
    plt.plot(source_train_acc, label="source_train_acc")
    plt.plot(source_val_acc, label="source_val_acc")
    plt.plot(source_test_acc, label="source_test_acc")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
