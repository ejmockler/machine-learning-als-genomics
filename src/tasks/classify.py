from typing import Iterable
from skopt.space import Space
from pandas import DataFrame
from tidyML import DataMediator


def optimizeModel(
    model, hyperparameterSpace: Iterable[Space], cases: DataFrame, controls: DataFrame
):
    from tidyML import BayesianOptimizer, RegressorCollection
    from pandas import concat

    def objectiveToMinimize(model, inputData, inputLabels, **parameters):
        import numpy as np
        from sklearn.model_selection import cross_val_score

        model.set_params(**parameters)
        return -np.mean(
            cross_val_score(
                model,
                inputData,
                inputLabels,
                cv=5,
                n_jobs=-1,
                scoring="neg_mean_absolute_error",
            )
        )

    regressors = RegressorCollection(GaussianProcess={})
    optimizer = BayesianOptimizer(regressors)

    numericLabels = [1] * len(cases) + [0] * len(controls)
    allData = concat([cases, controls])

    bestParameters, convergencePlot = optimizer.optimize(
        model,
        hyperparameterSpace,
        inputData=allData,
        inputLabels=numericLabels,
        objectiveToMinimize=objectiveToMinimize,
    )

    model.set_params(**bestParameters)
    return model, convergencePlot


def evaluateModel(model, dataset: DataMediator):
    results = {}
    for phase in ("train", "holdout-one", "holdout-two"):
        clonedModel = model.clone()
        if phase == "train":
            clonedModel.fit(dataset.train, dataset.trainLabels)
        elif "holdout" in phase:
            testData = getattr(dataset, "holdout" if "one" in phase else "validation")
            results[phase]["labelProbabilities"] = clonedModel.eval(testData)
    return results
