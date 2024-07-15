import numpy as np

### Custom Normalization functions

def NormStandard(dataset: np.array):
    mean = np.nanmean(dataset)
    std = np.nanstd(dataset)
    return (dataset - mean) / std

def NormMinMax(dataset: np.array):
    min = np.min(dataset)
    max = np.max(dataset)
    return (dataset - min) / (max - min)

def NormCustomBIS(dataset: np.array):
    return (100 - dataset) / 100

def NormNone(dataset: np.array):
    return dataset