import numpy as np
import pandas as pd

def Cross(data1,data2):
    diff = pd.DataFrame(data1 - data2)
    diff_shift = pd.DataFrame(diff.shift(1).values[:, 0])
    diff_shift.iloc[0] = diff.iloc[0]
    data = pd.DataFrame(np.zeros(len(data1)))
    data[(diff_shift < 0) & (diff > 0)] = 1
    return np.array(data.values[:,0])