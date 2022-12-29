import re

FILE_NAME = 'reformatted_main_data.log'


def reformat_main_data(file='abstract_main_data.log'):
    reformatted = []
    odd = []
    even = []
    f = open(file, 'r')
    lines = f.readlines()
    assert len(lines) % 2 == 0
    for i, line in enumerate(lines):
        if i % 2 == 0:
            even.append(line.strip('\n'))
        else:
            odd.append(re.sub("^[0-9-:, ]*", "", line.strip('\n')))
    for i in range(len(odd)):
        reformatted_line = even[i] + ', ' + odd[i] + '\n'
        reformatted.append(reformatted_line)
    _create_file(reformatted)
    f.close()


def _create_file(lines):
    f = open(FILE_NAME, 'w')
    for line in lines:
        f.write(line)
    f.close()


if __name__ == "__main__":
    reformat_main_data()
