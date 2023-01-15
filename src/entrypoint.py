from prefect import flow, task, unmapped
from prefect_dask.task_runners import DaskTaskRunner
from .tasks.load import textLinesToList, excelToDataframe, getBalancedSample
from .tasks.classify import optimizeModels

import hydra
from omegaconf import DictConfig, OmegaConf

from tidyML import NeptuneExperimentTracker

getControlIDs = task(textLinesToList)
getCaseIDs = task(textLinesToList)
getRelatedSampleIDs = task(textLinesToList)

getSampleMetadata = task(excelToDataframe())


@hydra.main(version_base=None, config_path="conf", config_name="config")
def start(configToInitialize: DictConfig):
    config = OmegaConf.to_yaml(configToInitialize)

    @flow(task_runner=DaskTaskRunner)
    def linkInput():
        relatedSampleIDs = getRelatedSampleIDs(config.filePaths["relatedSamples"])
        return {
            "controlIDs": getControlIDs(
                config.filePaths["controlIDs"], relatedSampleIDs
            ),
            "caseIDs": getCaseIDs(config.filePaths["caseIDs"], relatedSampleIDs),
            "sampleMetadata": getSampleMetadata(config.filePaths["sampleMetadata"]),
        }

    inputs = linkInput()

    @flow(task_runner=DaskTaskRunner)
    def sampleData(iterations=config.sampling["bootstrapIterations"]):
        import random

        experimentTracker = NeptuneExperimentTracker(
            config.tracker["projectID"],
            config.tracker["entityID"],
            config.tracker["analysisName"],
        )
        random.seed(config.sampling["randomSeed"])

        bootstrapDatasets = getBalancedSample.map(
            unmapped(inputs),
            unmapped(config),
            [random.randint() for i in config.sampling["bootstrapIterations"]],
        )

        optimizeModels(config.model.classList, bootstrapDatasets[0])


if __name__ == "__main__":
    start()
