import numpy as np

def maxstar(x_array):
    if x_array.shape[0] == 0:
        return 0
    m = max(x_array)
    return m + np.log(np.sum(np.exp(x_array-m)))

def maxstar_cte(x_array):
    if x_array.shape[0] == 0:
        return 0
    m = max(x_array)
    j_array = np.zeros(x_array.shape)
    for i in range(j_array.shape[0]):
        if np.abs(x_array[i]-m) < 1.61:
            j_array[i] = 0.5
    return m + np.average(j_array)