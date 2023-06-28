import numpy as np

def mape_loss(pred, targets):
    abs_error     = (np.abs(targets - pred)) / targets
    sum_abs_error = np.sum(abs_error)
    mape_loss     = (sum_abs_error / targets.size) * 100
    return mape_loss

