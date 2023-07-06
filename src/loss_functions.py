import numpy as np

def ape_loss(pred, targets):
    '''
    Absolute Percent Error
    
    '''
    assert pred.shape == targets.shape, "Predictions and Targets must be same shape"
    abs_error     = (np.abs(targets - pred)) / targets
    sum_abs_error = np.sum(abs_error)
    # ape_loss     = (sum_abs_error / targets.size) * 100
    return sum_abs_error

