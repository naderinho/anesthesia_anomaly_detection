import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import modules as mod

tracks = [
    'Orchestra/PPF20_VOL'
]

samples = mod.get_vitaldb_samples(case_id=3570, track_names=tracks)

pd.DataFrame(samples).to_csv('data/samples.csv')
print(samples.shape)

plt.plot(samples,'.')
plt.show()