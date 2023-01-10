import os
import re

from generate_visual_graph import generate_visual_graph
from utils.get_field_data import get_field_data

FEAT = 'FEAT'
ACC = 'ACC'
LINEAR_CKA = 'linear_cka'
KERNEL_CKA_S = 'kernel_cka_s'


def main(rootLocation):
    targets = [FEAT, ACC]
    factors = [LINEAR_CKA, KERNEL_CKA_S]
    toAnalysisFolders = [folder for folder in os.listdir(rootLocation) if os.path.isdir(os.path.join(rootLocation, folder)) if folder != 'data_visualization']
    testLogs = []
    modelNames = []
    for folder in toAnalysisFolders:
        testLogFileName = 'test.log'
        files = os.listdir(rootLocation + '/' + folder)
        files.index(testLogFileName)  # throw an error if it does not exist
        testLogs.append(rootLocation + '/' + folder + '/' + testLogFileName)
        modelNames.append(_get_model_name(folder))

    for testLog in testLogs:
        for i in range(len(targets)):
            targetData = get_field_data(testLog, targets[i])
            factorsData = [get_field_data(testLog, factor) for factor in factors]
            generate_visual_graph(targetData, factorsData, modelNames[i])


def _get_model_name(folderName):
    prefix = 'Meta-feat-acc-acm-'
    modelName = re.search('^[A-Z]+', re.sub(prefix, "", folderName)).group()
    return modelName


if __name__ == '__main__':
    main("..")
