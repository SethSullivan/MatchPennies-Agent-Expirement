import numpy as np

def ape_loss(pred: np.ndarray, targets: np.ndarray, ):
    '''
    Absolute Percent Error
    
    '''
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(targets, list):
        targets = np.array(targets)
    assert pred.shape == targets.shape, "Predictions and Targets must be same shape"
    pred = pred.squeeze()
    targets = targets.squeeze()
    # if drop_condition_num is not None:
    #     if pred.ndim>1:
    #         pred = np.delete(pred, drop_condition_num, 1)
    #         targets = np.delete(targets, drop_condition_num, 1)
    #     else:
    #         pred = np.delete(pred, drop_condition_num, 0)
    #         targets = np.delete(targets, drop_condition_num, 0)
        
    abs_error     = (targets - pred)**2 / targets**2
    sum_abs_error = np.sum(abs_error)
    # ape_loss     = (sum_abs_error / targets.size) * 100
    return sum_abs_error

