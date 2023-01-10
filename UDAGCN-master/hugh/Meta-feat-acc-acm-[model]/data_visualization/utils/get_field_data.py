import re


def get_field_data(fileLocation, fieldName):
    f = open(fileLocation, 'r')
    content = f.read()
    matched = re.findall("{}\s*=\s*[0-9.]+".format(fieldName), content)
    f.close()
    return {"field": fieldName, "info": list(map(_abstract_numbers, matched))}


def _abstract_numbers(data):
    return float(re.sub("[^\d.]", "", data))
