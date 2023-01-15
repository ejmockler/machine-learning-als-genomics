def textLinesToList(textFilePath, linesToExclude: list[str]):
    lineList = []
    with open(textFilePath) as file:
        for line in file:
            if line not in linesToExclude:
                lineList.append(line)
    return lineList


def excelToDataframe(excelFilePath):
    import pandas as pd

    return pd.read_excel(excelFilePath)


def getBalancedSample(inputs, config, randomSeed):
    from tidyML import DataMediator
    import pandas as pd

    return DataMediator(
        pd.DataFrame(config.filePaths["variantCalls"]),
        config.sampling["testProportion"],
        inputs["controlIDs"],
        inputs["caseIDs"],
        randomSeed,
        balancingMethod="downsampling",
    )
