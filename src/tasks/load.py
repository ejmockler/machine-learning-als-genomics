def textLinesToList(textFilePath, linesToExclude: list[str]):
    lineList = []
    with open(textFilePath) as file:
        for line in file:
            if line not in linesToExclude:
                lineList.append(line)
    return lineList


def excelToDataFrame(excelFilePath):
    import pandas as pd

    return pd.read_excel(excelFilePath)


def embedSamples(variantCallPath, testProportion, controlIDs, caseIDs, randomSeed):
    from tidyML import DataMediator
    import pandas as pd

    return DataMediator(
        pd.DataFrame(variantCallPath, sep="/t"),
        testProportion,
        controlIDs,
        caseIDs,
        randomSeed,
        balancingMethod="downsampling",
    )


def prepareClassificationDataset(samples):
    pass
