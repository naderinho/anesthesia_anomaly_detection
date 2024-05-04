import pandas as pd


from .relevant_tracks import relevant_track_names

def vital_filter_clinical_info():
    """extracts the relevant columns from the cases.csv file and applies one-hot encoding to text columns
    Returns:
        pandas.df
    """
    df = pd.read_csv('data/cases.csv')

    # Select relevant columns to be used
    relevant_columns = [
        'caseid',
        'age',
        'sex',
        'height',
        'weight',
        'bmi',
        'ane_type',
        'asa'
    ]
    df = df[relevant_columns]

    ### Encode Gender
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)

    ### Collect only general anesthesia without spinal and sedationalgesia
    df = df[df['ane_type'] == 'General'].drop(columns=['ane_type'])

    return df

def vital_filter_tracks():
    """Extracts the caseids of the cases that have all the required tracks
    listed in the track_names variable in relevant_tracks.py

    Returns:
        list: caseids
    """

    df = pd.read_csv('data/tracks_sorted.csv')

    return list(df['caseid'].loc[df[relevant_track_names].all(1)])

def vital_select_cases():
    """Combines the clinical information and the caseids of the cases that have 
    all the required tracks in a single dataframe

    Returns:
        pd.DataFrame: _description_
    """
    clinical_info = vital_filter_clinical_info()
    clinical_info = clinical_info[clinical_info['caseid'].isin(vital_filter_tracks())]

    return clinical_info.reset_index(drop=True)