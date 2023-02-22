from pandas import DataFrame


def textLinesToList(textFilePath: str, linesToExclude: list[str]):
    lineList = []
    with open(textFilePath) as file:
        for line in file:
            if line not in linesToExclude:
                lineList.append(line)
    return lineList


def excelToDataFrame(excelFilePath: str):
    import pandas as pd

    return pd.read_excel(excelFilePath)


def vcfLikeToDataFrame(clinicalMetadata: DataFrame, vcfLikePath: str):
    # read VCF as dataframe
    # if headers
        # extract fields
        # slice genotype table
    # else
        # verify VCF fields & format
    # genotypes = vcf[clinicalMetadata[sampleIDs]]
    # chromosomePositionMetadata = vcf[fieldIndices]
    
    # TODO check whether genotype metadata exists & can be returned
    # return chromosomePositionMetadata, genotypes
    pass


def embedSamples(vcfDataFrame, testProportion, controlIDs, caseIDs, randomSeed):
    from tidyML import DataMediator

    return DataMediator(
        vcfDataFrame,
        testProportion,
        controlIDs,
        caseIDs,
        randomSeed,
        balancingMethod="downsampling",
    )


def prepareClassificationDataset(samples):
    pass
