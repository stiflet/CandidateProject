import numpy as np

def calculate_crps(preds:np.array, thresholds:np.array, y):
    # Making a matrix with the cdf function for every model run.
    cdf_matrix = preds.reshape(-1, len(thresholds))
    
    # Making a target matrix for the binary target observation for each model run.
    target_matrix = y.to_numpy().reshape(-1, len(thresholds))
    
    # Calculating CRPS.
    # Where np.trapezoid is the integral.
    crps = np.trapezoid((cdf_matrix - target_matrix)**2, thresholds, axis=1)
    
    return crps.mean()