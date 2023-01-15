from prefect import flow, task, unmapped
from prefect_dask.task_runners import DaskTaskRunner
from .tasks.load import textLinesToList, excelToDataframe, getBalancedSample
from .tasks.classify import optimizeModel, evaluateModel, trackResult

import hydra
from omegaconf import DictConfig, OmegaConf

from tidyML import NeptuneExperimentTracker

getControlIDs = task(textLinesToList)
getCaseIDs = task(textLinesToList)
getRelatedSampleIDs = task(textLinesToList)
getSampleMetadata = task(excelToDataframe())

getBalancedSample = task(getBalancedSample)

optimizeModel = task(optimizeModel)


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

        random.seed(config.sampling["randomSeed"])

        bootstrapDatasets = getBalancedSample.map(
            unmapped(config.filePaths["variantCalls"]),
            unmapped(config.sampling["testProportion"]),
            inputs["controlIDs"],
            inputs["caseIDs"],
            [random.randint() for i in iterations],
        )

        models = optimizeModel.map(
            config.model.classList, unmapped(bootstrapDatasets[0])
        )

        return bootstrapDatasets, models

    bootstrapDatasets, optimizedModels = sampleData()

    @flow(task_runner=DaskTaskRunner)
    def classifySamples():
        experimentTracker = NeptuneExperimentTracker(
            config.tracker["projectID"],
            config.tracker["entityID"],
            config.tracker["analysisName"],
        )

        @task
        def run(dataset):
            for model in optimizedModels:
                results = evaluateModel(model, dataset)
                trackResult.map(results.keys(), results.values(), experimentTracker)

        run.map(bootstrapDatasets)

    classifySamples()


if __name__ == "__main__":
    start()
