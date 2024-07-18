"""
This module contains functions to normalize the input dataset.
"""

import numpy as np

def NormStandard(dataset: np.array) -> np.array:
    """
    Normalizes the input dataset to a mean of 0 and a standard deviation of 1.

    Args:
        dataset (np.array): Input dataset

    Returns:
        np.array: Normalized dataset
    """

    mean = np.nanmean(dataset)
    std = np.nanstd(dataset)
    return (dataset - mean) / std

def NormMinMax(dataset: np.array) -> np.array:
    """
    Normalizes the input dataset to a range between 0 and 1.

    Args:
        dataset (np.array): Input dataset

    Returns:
        np.array: Normalized dataset
    """
    min = np.min(dataset)
    max = np.max(dataset)
    return (dataset - min) / (max - min)

def NormCustomBIS(dataset: np.array) -> np.array:
    """
    Normalizes the Bispectral input dataset to a value between 0 and 1, where 0 
    correspond to a BIS value of 100 and 1 to a BIS value of 0.

    Args:
        dataset (np.array): Input dataset

    Returns:
        np.array: Normalized dataset
    """
    return (100 - dataset) / 100

def NormNone(dataset: np.array) -> np.array:
    """
    Applies no normalization to the input dataset.

    Args:
        dataset (np.array): Input dataset

    Returns:
        np.array: Normalized dataset
    """
    return dataset