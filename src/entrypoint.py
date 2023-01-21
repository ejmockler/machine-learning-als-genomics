from prefect import flow, task, unmapped
from prefect.task_runners import ConcurrentTaskRunner
from prefect_dask.task_runners import DaskTaskRunner

from .tasks.load import textLinesToList, excelToDataFrame, embedSamples
from .tasks.classify import optimizeModel, evaluateModel
from .tasks.track import trackResult

import hydra
from omegaconf import DictConfig, OmegaConf

from ..conf.models import stack

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

    @flow(task_runner=ConcurrentTaskRunner)
    def linkInput():
        relatedSampleIDs = getRelatedSampleIDs(config.input["relatedSampleIDs"])
        return {
            "controlIDs": getControlIDs(
                config.input["controlIDs"], relatedSampleIDs
            ).submit(),
            "caseIDs": getCaseIDs(config.input["caseIDs"], relatedSampleIDs).submit(),
            "sampleMetadata": getSampleMetadata(
                config.input["sampleMetadata"]
            ).submit(),
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
        ).submit()
        return (bootstrapDatasets,)

    @flow(task_runner=DaskTaskRunner)
    def classifySamples():
        from tidyML import DataMediator, NeptuneExperimentTracker

        bootstrapDatasets, optimizedModels = sampleData()

        optimizedModels = optimizeModel.map(
            stack,
            cases=unmapped(
                bootstrapDatasets.dataframe.iloc[
                    bootstrapDatasets.originalExperimentalIDs
                ]
            ),
            controls=unmapped(
                bootstrapDatasets.dataframe.iloc[bootstrapDatasets.originalControlIDs]
            ),
        )

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
                trackResult(results.keys(), results.values(), experimentTracker)

        runSample.map(bootstrapDatasets).submit()

    classifySamples()


if __name__ == "__main__":
    start()
