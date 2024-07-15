import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

def get_anaesthesia_phases(dataset: np.array, N: int = 60):
    """
    A function to detect the anesthesia phases in a given dataset. The function calculates the
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

def phases_report(prediction: np.array, groundtruth: np.array, propofolrate: np.array):

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