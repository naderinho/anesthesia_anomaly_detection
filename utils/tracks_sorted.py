"""
This script reads the trks.csv file and creates a new file tracks_sorted.csv which 
contains the caseids and the tracks that are present in the case.
"""

import pandas as pd


df = pd.read_csv('data/trks.csv')
tnames = list(df['tname'].unique())

columns = ['caseid']
columns += tnames

tracks = pd.DataFrame(columns=columns)

for i in df['caseid'].unique():
    row = [i]
    df_case = df[df['caseid'] == i]['tname']
    for j in tnames:
        if j in df_case.values:
            row.append(True)
        else:
            row.append(False)

    tracks.loc[len(tracks.index)] = row

tracks.to_csv('data/tracks_sorted.csv', index=False)