def textLinesToList(textFilePath: str):
    lineList = []
    with open(textFilePath) as file:
        for line in file:
            lineList.append(line)
    return lineList
