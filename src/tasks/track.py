from tidyML import NeptuneExperimentTracker


def trackResult(
    path, keys, values, tracker: NeptuneExperimentTracker, projectLevel=False
):
    if len(keys) != len(values):
        raise ValueError(
            f"Number of keys {len(keys)} does not equal number of values {len(values)}"
        )
    tracker.log(path, {key: value for key, value in zip(keys, values)}, projectLevel)
