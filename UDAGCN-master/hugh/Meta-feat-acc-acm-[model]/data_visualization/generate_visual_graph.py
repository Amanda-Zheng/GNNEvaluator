from matplotlib import pyplot as plt

from utils.scatter import compare_fields_data as compare_fields_data_in_scatter


def generate_visual_graph(targetData, factorsData, title):
    targetField = targetData["field"]
    targetInfo = targetData["info"]
    factorsField = [factor["field"] for factor in factorsData]
    factorsInfo = [factor["info"] for factor in factorsData]

    _validate(targetInfo, factorsInfo)
    compare_fields_data_scatter_container = compare_fields_data_in_scatter(factorsField[0], factorsInfo[0], targetField, targetInfo)
    for i in range(len(factorsInfo[1:-1])):
        compare_fields_data_scatter_container(factorsField[i + 1], factorsInfo[i + 1], targetField, targetInfo)
    compare_fields_data_scatter_container(factorsField[-1], factorsInfo[-1], targetField, targetInfo, done=True)

    plt.title(title)
    plt.ylabel(targetField)
    plt.legend()
    plt.show()


def _validate(targetInfo, factorsInfo):
    for factorData in factorsInfo:
        assert len(factorData) == len(targetInfo)


def _deconstruct_data(data):
    return data.field, data.info
