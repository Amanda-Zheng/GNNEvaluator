import os
import re

import matplotlib.pyplot as plt

from utils.get_field_data import get_field_data
from utils.scatter.compare_fields_data import compare_fields_data

SORTED_METHODS_PATH = '../data/methods/sorted'
RATE_FIELD_SUFFIX = '_rate'
TARGET_FIELD_NAMES = ['FEAT', 'ACC']
SPECIAL_METHOD_NAME = 'combo'


def generate_field_rate_with_target_field_visualization(targetFieldNames):
    methodFiles = os.listdir(SORTED_METHODS_PATH)
    methodNames = remove_extension(methodFiles)
    for targetFieldName in targetFieldNames:
        for i, methodName in enumerate(methodNames):
            methodFieldRateName = _get_method_field_rate_name(methodName)
            filePath = SORTED_METHODS_PATH + '/' + methodFiles[i]
            targetField = get_field_data(targetFieldName, filePath)
            plt.ylabel(targetFieldName, color='b')
            if not _is_special_method(methodName):
                fieldRates = get_field_data(methodFieldRateName, filePath)
                plt.xlabel(methodFieldRateName, color='b')
                compare_fields_data(fieldRates, targetField, done=True)
            else:
                otherMethodNames = remove_item(methodNames, SPECIAL_METHOD_NAME)
                fieldRatesList = [get_field_data(_get_method_field_rate_name(methodName), filePath) for methodName in otherMethodNames]
                if len(otherMethodNames) > 1:
                    annotate_label_with_arrow(_get_method_field_rate_name(otherMethodNames[0]), (fieldRatesList[0][0], targetField[0]))
                    compareFieldsData = compare_fields_data(fieldRatesList[0], targetField)
                    for i, fieldRates in enumerate(fieldRatesList[1:-1]):
                        annotate_label_with_arrow(_get_method_field_rate_name(otherMethodNames[i + 1]), (fieldRatesList[i + 1][0], targetField[0]))
                        compareFieldsData(fieldRates, targetField)
                    annotate_label_with_arrow(_get_method_field_rate_name(otherMethodNames[-1]), (fieldRatesList[-1][0], targetField[0]))
                    compareFieldsData(fieldRatesList[-1], targetField, done=True)
                else:
                    plt.xlabel(_get_method_field_rate_name(otherMethodNames[0]))
                    compareFieldsData(fieldRatesList[0], targetField, done=True)


def remove_extension(fileNames):
    def replace(fileName):
        return re.sub('\..*', "", fileName)

    return list(map(replace, fileNames))


def remove_item(list, item):
    pendingList = list.copy()
    pendingList.remove(item)
    return pendingList


def annotate_label_with_arrow(label, location):
    (x, y) = location
    plt.annotate(label,
                 xy=location, xycoords='data',
                 xytext=(x, y + 0.1), textcoords='data',
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3"),
                 )


def _is_special_method(fieldName):
    return fieldName == SPECIAL_METHOD_NAME


def _get_method_field_rate_name(methodName):
    return methodName + RATE_FIELD_SUFFIX


if __name__ == "__main__":
    generate_field_rate_with_target_field_visualization(TARGET_FIELD_NAMES)
