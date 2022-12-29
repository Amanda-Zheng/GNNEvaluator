import re

FILE_LOCATION = '../format_data/reformatted_main_data.log'


def filter_methods(file=FILE_LOCATION):
    methods = {}
    f = open(file, 'r')
    lines = f.readlines()
    for i, line in enumerate(lines):
        methodName = re.search('".*"-method', line).group(0)
        if methodName not in methods:
            methods[methodName] = []
        methods[methodName].append(line)
    for key in methods.keys():
        fileLocation = re.search('".*"', key).group(0).replace('"', "") + '.log'
        f = open(fileLocation, 'w')
        for line in methods[key]:
            f.write(line)
        f.close()


if __name__ == "__main__":
    filter_methods()
