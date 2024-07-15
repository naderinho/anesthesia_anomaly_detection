from scipy import ndimage
import pandas as pd
import numpy as np

def outlierfilter(data: pd.DataFrame,threshhold: float, iterations: int, min: float, max: float):
    """
    A filter function, which calculates the gradient of a given Pandas DataFram Timeseries
    and performs a binary dilation on datapoints which exceed a certain treshhold, to detect
    and remove unwanted outliers in the dataset. Additionally all values exceeding a given
    min/max value are replaced with np.nan and linearly interpolated with the Pandas interpolate
    method.

    Args:
        data (pd.DataFrame): Timeseries Data
        threshhold (float): Gradient thresshold
        iterations (int): number of iterations of the binary dilation
        min (float): maximum expected value
        max (float): minimum expected value

    Returns:
        pd.DataFrame: _description_
    """
    gradient = np.diff(data,n=1, axis=0, append=0)
    gradientfilter = ndimage.binary_dilation(np.abs(gradient) > threshhold, iterations=iterations)

    # Apply Filter
    data[gradientfilter] = np.nan

    data[data <= min] = np.nan
    data[data > max] = np.nan

    data = data.interpolate(method = 'linear')
    data = data.bfill()
    return data