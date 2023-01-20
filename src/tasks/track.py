from tidyML import NeptuneExperimentTracker
def trackResult(keys, values, tracker: NeptuneExperimentTracker):
    if len(keys) != len(values):
        raise ValueError(f"Number of keys {len(keys)} does not equal number of values {len(values)}")
    tracker.log({key: value for key, value in zip(keys,values)})
