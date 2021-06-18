import numpy as np
def ts_dot(x, y):
    ''' dot product of two time series (is there a better way?) '''
    z = np.zeros(np.shape(x)[1])
    for x1, y1 in zip(x, y):
        z += x1 * y1
    return z

def ts_dot_hat(x, yhat):
    ''' dot product of time series w/ const vec '''
    z = np.zeros(np.shape(x)[1])
    for idx, x1 in enumerate(x):
        z += x1 * yhat[idx]
    return z

def ts_dot_uv(x, y):
    ''' dot product of two time series (is there a better way?) '''
    z = np.zeros(np.shape(x)[1])
    for x1, y1 in zip(x, y):
        z += x1 * y1
    return z / np.sqrt(np.sum(x**2, axis=0) * np.sum(y**2, axis=0))
