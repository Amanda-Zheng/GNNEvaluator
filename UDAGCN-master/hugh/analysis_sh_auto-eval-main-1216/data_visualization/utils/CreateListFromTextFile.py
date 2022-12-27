def create_list_from_text_file(fileLocation):
    file = open(fileLocation, "r")
    linesList = []
    for line in file:
        strippedLine = line.strip()
        linesList.append(strippedLine)
    file.close()
    return linesList
