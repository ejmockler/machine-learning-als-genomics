from prefect import flow, task, unmapped
from prefect_dask.task_runners import DaskTaskRunner

from .tasks.load import textLinesToList, excelToDataFrame, embedSamples
from .tasks.classify import optimizeModel, evaluateModel
from .tasks.track import trackResult

import hydra
from omegaconf import DictConfig, OmegaConf

getControlIDs = task(textLinesToList)
getCaseIDs = task(textLinesToList)
getRelatedSampleIDs = task(textLinesToList)
getSampleMetadata = task(excelToDataFrame)

embedSamples = task(embedSamples)

optimizeModel = task(optimizeModel)
evaluateModel = task(evaluateModel)

trackResult = task(trackResult)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def start(configToInitialize: DictConfig):
    config = OmegaConf.to_yaml(configToInitialize)

    @flow(task_runner=DaskTaskRunner)
    def linkInput():
        relatedSampleIDs = getRelatedSampleIDs(config.input["relatedSampleIDs"])
        return {
            "controlIDs": getControlIDs(config.input["controlIDs"], relatedSampleIDs),
            "caseIDs": getCaseIDs(config.input["caseIDs"], relatedSampleIDs),
            "sampleMetadata": getSampleMetadata(config.input["sampleMetadata"]),
        }

    @flow(task_runner=DaskTaskRunner)
    def sampleData(iterations=config.sampling["bootstrapIterations"]):
        import random

        inputs = linkInput()

        random.seed(config.sampling["randomSeed"])

        bootstrapDatasets = embedSamples.map(
            unmapped(config.input["variantCalls"]),
            unmapped(config.sampling["testProportion"]),
            inputs["controlIDs"],
            inputs["caseIDs"],
            [random.randint() for i in iterations],
        )

        models = optimizeModel.map(
            config.model.classList, unmapped(bootstrapDatasets[0])
        )

        return bootstrapDatasets, models

    @flow(task_runner=DaskTaskRunner)
    def classifySamples():
        from tidyML import DataMediator, NeptuneExperimentTracker

        bootstrapDatasets, optimizedModels = sampleData()

        experimentTracker = NeptuneExperimentTracker(
            config.tracker["project"],
            config.tracker["entity"],
            config.tracker["analysis"],
        )

        @task
        def runSample(dataset: DataMediator):
            dataset.resample()
            for model in optimizedModels:
                results = evaluateModel(model, dataset)
                trackResult.map(results.keys(), results.values(), experimentTracker)

        runSample.map(bootstrapDatasets)

    classifySamples()


if __name__ == "__main__":
    start()
