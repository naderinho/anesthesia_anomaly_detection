import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import modules as mod

cases = mod.vital_select_cases()

print(len(cases.index))