"""
This module contains functions to evaluate the performance of a trained model.
"""

import numpy as np
import pandas as pd
import scipy.stats
pd.options.display.float_format = '{:,.2f}'.format

def get_anaesthesia_phases(dataset: np.array, N: int = 60) -> dict:
    """
    A function to detect the anesthesia phases based on TIVA in a given dataset. The function calculates the
    moving average of the propofol rate and detects the start and end of the anesthesia phases
    by detecting the first and last occurence of a propofol rate below 40 ml/h and above 5 ml/h.

    Args:
        dataset (np.array): Propofol rate dataset
        N (int): Number of samples to average

    Returns:
        dict: Dict where each key contains a List of the end index of induction and start index of the recovery phases
    """
    anesthesia_phases = []
    
    for data in dataset:
        conv = np.convolve(data[:,0], np.ones(N)/N, mode='valid')
        index1 = np.argmax(conv < 40) + 2 * N
        index2 = np.where(conv > 5)[0][-1] - N
        anesthesia_phases.append([index1, index2])

    return anesthesia_phases

def get_anaesthesia_phases_balanced(bis_dataset: np.array ,mac_dataset: np.array, N: int = 60) -> dict:
    """
    A function to detect the anesthesia phases based on a balanced anaesthesia in a given dataset. The function calculates the
    moving average of the propofol rate and detects the start and end of the anesthesia phases
    by detecting the first and last occurence of a propofol rate below 40 ml/h and above 5 ml/h.

    Args:
        dataset (np.array): Propofol rate dataset
        N (int): Number of samples to average

    Returns:
        dict: Dict where each key contains a List of the end index of induction and start index of the recovery phases
    """
    anesthesia_phases = []
    
    for bis, mac in zip(bis_dataset, mac_dataset):
        conv_bis = np.convolve(bis[:,0], np.ones(N)/N, mode='valid')
        conv_mac = np.convolve(mac[:,0], np.ones(N)/N, mode='valid')
        index1 = np.argmax(conv_bis < 60) + 2 * N
        index2 = np.where(conv_mac > 0.5)[0][-1] - N
        anesthesia_phases.append([index1, index2])

    return anesthesia_phases

def phases_report(prediction: np.array, groundtruth: np.array, propofolrate: np.array) -> pd.DataFrame:
    """
    From a given BIS prediction with the corresponding groundtruth values, this function
    calculates the MSE, MAE and RMSE for the whole dataset and the three anesthesia phases
    (Induction, Maintenance, Recovery). The function also calculates the same metrics for a
    baseline model that always predicts a BIS value of 41.0.

    Args:
        prediction (np.array): predicted BIS values
        groundtruth (np.array): measured BIS values
        propofolrate (np.array): infusion rate of propofol 20mg/ml in ml/h

    Returns:
        pd.DataFrame: with the calculated metrics for the whole dataset and the three anesthesia phases
    """

    baseline = np.ones(groundtruth.shape) * 41.0

    phases = get_anaesthesia_phases(dataset = propofolrate)

    # Create the three datasets
    all_pred, induction_pred, maintenance_pred, recovery_pred = np.copy(prediction), np.copy(prediction), np.copy(prediction), np.copy(prediction)
    all_base, induction_base, maintenance_base, recovery_base = np.copy(baseline), np.copy(baseline), np.copy(baseline), np.copy(baseline)

    all_pred[groundtruth == 0.0] = np.nan
    all_base[groundtruth == 0.0] = np.nan

    for i, phase in enumerate(phases):
        induction_pred[i,phase[0]:-1,:] = np.nan
        maintenance_pred[i,0:phase[0],:] = np.nan
        maintenance_pred[i,phase[1]:-1,:] = np.nan
        recovery_pred[i,0:phase[1],:] = np.nan
        recovery_pred[i,np.where(groundtruth[i] == 0)[0][0]:-1,:] = np.nan

        induction_base[i,phase[0]:-1,:] = np.nan
        maintenance_base[i,0:phase[0],:] = np.nan
        maintenance_base[i,phase[1]:-1,:] = np.nan
        recovery_base[i,0:phase[1],:] = np.nan
        recovery_base[i,np.where(groundtruth[i] == 0)[0][0]:-1,:] = np.nan

    table_index = ['All', 'Induction', 'Maintenance', 'Recovery']

    results = pd.DataFrame(index=table_index, columns=['Prediction MSE', 'Baseline MSE', 'Prediction MAE', 'Baseline MAE', 'Prediction RMSE', 'Baseline RMSE'])

    for i, section in enumerate([[all_pred, all_base], [induction_pred, induction_base], [maintenance_pred, maintenance_base], [recovery_pred, recovery_base]]):
        results.loc[table_index[i]] = [
            np.nanmean(np.square(groundtruth - section[0])),           # MSE Prediction
            np.nanmean(np.square(groundtruth - section[1])),           # MSE Baseline
            np.nanmean(np.abs(groundtruth - section[0])),              # MAE Prediction
            np.nanmean(np.abs(groundtruth - section[1])),              # MAE Baseline
            np.sqrt(np.nanmean(np.square(section[0] - groundtruth))),   # RMSE Prediction
            np.sqrt(np.nanmean(np.square(section[1] - groundtruth)))    # RMSE Baseline
        ]

    return results

def MSE_phases(prediction: np.array, groundtruth: np.array, propofolrate: np.array) -> pd.DataFrame:

    phases = get_anaesthesia_phases(dataset = propofolrate)

    # Create the three datasets
    all_pred, induction_pred, maintenance_pred, recovery_pred = np.copy(prediction), np.copy(prediction), np.copy(prediction), np.copy(prediction)
    all_pred[groundtruth == 0.0] = np.nan

    for i, phase in enumerate(phases):
        induction_pred[i,phase[0]:-1,:] = np.nan
        maintenance_pred[i,0:phase[0],:] = np.nan
        maintenance_pred[i,phase[1]:-1,:] = np.nan
        recovery_pred[i,0:phase[1],:] = np.nan
        recovery_pred[i,np.where(groundtruth[i] == 0)[0][0]:-1,:] = np.nan


    table_index = ['All', 'Induction', 'Maintenance', 'Recovery']

    results = pd.DataFrame(index=table_index, columns=['Prediction MSE'])

    for i, section in enumerate([all_pred, induction_pred, maintenance_pred, recovery_pred]):
        results.loc[table_index[i]] = [
            np.nanmean(np.square(groundtruth - section)),           # MSE Prediction
        ]

    return results

def phases_report_std(report: pd.DataFrame, prediction: np.array, groundtruth: np.array, propofolrate: np.array) -> pd.DataFrame:
    sets = prediction.shape[0]

    evaluation = np.zeros((sets,4,6))

    for i in range(0,sets):
        evaluation[i] = phases_report(prediction[i:i+1], groundtruth[i:i+1], propofolrate[i:i+1])

    # Prediction RMSE min/max (5)

    table_index = ['All     ', 'Induction', 'Maintenance', 'Recovery']

    for j, phase in enumerate(table_index):
        print(phase, '\tmin: \t', np.argmin(evaluation[:,j,5]), '\tmax: \t', np.argmax(evaluation[:,j,5])),'(', '{:.3f}'.format(np.max(evaluation[:,j,5]))

    data = np.std(evaluation, axis=0)
    return pd.DataFrame(data, index=report.index, columns=report.columns)