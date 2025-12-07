import numpy as np

def tanh_segm1(x_array):
    return np.array([np.sign(x) if (np.abs(x)>1) else x for x in x_array])

def tanh_segm2(x_array):
    return np.array([np.sign(x) if (np.abs(x)>1.5) else (0.5*x+0.25*np.sign(x) if (np.abs(x)>0.5) else x) for x in x_array])

def tanh_segm3(x_array):
    return np.array([np.sign(x) if (np.abs(x)>2) else (0.25*x+0.5*np.sign(x) if (np.abs(x)>1) else (0.5*x+0.25*np.sign(x) if (np.abs(x)>0.5) else x)) for x in x_array])