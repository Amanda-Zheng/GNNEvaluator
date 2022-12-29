import re


def get_field_data(fieldName, fileLocation):
    f = open(fileLocation, 'r')
    content = f.read()
    matched = re.findall("{} = [0-9.]*".format(fieldName), content)
    f.close()
    return list(map(_abstract_info, matched))


def _abstract_info(data):
    abstracted = re.search("[0-9.]+", data).group(0)
    return float(abstracted)


