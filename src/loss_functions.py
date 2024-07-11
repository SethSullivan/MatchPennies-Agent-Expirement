import numpy as np

def ape_loss(pred: np.ndarray, targets: np.ndarray, ):
    '''
    12/07/23 -   Absolute Percent Error, ended up just doing the sum of the absolute error
    07/10/24 - Used this primarily, but going to try mean squared error 
    
    '''
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(targets, list):
        targets = np.array(targets)
    assert pred.shape == targets.shape, "Predictions and Targets must be same shape"
    pred = pred.squeeze()
    targets = targets.squeeze()
    
    abs_error     = np.abs((targets - pred) / targets)
    sum_abs_error = np.sum(abs_error)
    # ape_loss     = (sum_abs_error / targets.size) * 100
    return sum_abs_error

def mse_loss(predictions, targets):
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(targets, list):
        targets = np.array(targets)
        
    assert predictions.shape == targets.shape, "Predictions and Targets must be same shape"
    
    predictions = predictions.squeeze()
    targets = targets.squeeze()
    
    error = np.mean((predictions - targets)**2)
    
    return error 
