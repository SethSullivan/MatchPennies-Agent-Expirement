import numpy as np
from numba import njit
import numba as nb
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# @njit(parallel = True)
def pairwise_bootstrap(data_list, M = int(1e4), paired = False, alternative = "two-sided"):
    rng = np.random
    n = len(data_list)
    original_mean_diffs = np.zeros((n,n),np.float64)*np.nan
    original_mean_diff = 10
    results = np.zeros((n,n,M),np.float64)*np.nan
    comparison_tracker = []
    # GEt original, non-bootstrapped mean diffs
    for i in range(n):
        for j in range(n):
            if i+j in comparison_tracker or i == j:
                continue
            else:
                original_mean_diffs[i,j]= np.nanmean(data_list[i]) - np.nanmean(data_list[j])
                comparison_tracker.append(i+j)
    
    if paired:
        comparison_tracker = []
        for i,data1 in enumerate(data_list):
            for j,data2 in enumerate(data_list):
                if i+j in comparison_tracker or i==j:
                    continue
                else:
                    comparison_tracker.append(i+j)
                    assert data1.shape == data2.shape
                    #Want to resample from the distribution of paired differences with replacement
                    paired_diff = data1 - data2
                    data_len = paired_diff.shape[0]
                    for k in nb.prange(M):
                        results[i,j,k] = np.nanmean(rng.choice(paired_diff, size = data_len, replace = True))
    else:        
        data_len = data1.shape[0]
        
        #create a bucket with all data thrown inside
        pooled_data = np.empty(data1.shape[0] + data2.shape[0]) * np.nan
        pooled_data[:data1.shape[0]] = data1
        pooled_data[data1.shape[0]:] = data2
        
        #Recreate the two groups by sampling without replacement
        for i in nb.prange(M): 
            tmp = rng.choice(pooled_data, size = len(pooled_data), replace = False)
            data1_resample = tmp[:data_len] #up to number of points in data1
            data2_resample = tmp[data_len:] #the rest are in data2
            mean_diff = np.nanmean(data1_resample) - np.nanmean(data2_resample)
            results[i] = mean_diff
        
    #center the results on 0
    centered_results = results - np.nanmean(results)

    if alternative == "two-sided":
        #are the results more extreme than the original?
        p_val = np.sum(centered_results > abs(original_mean_diff),axis=2) + np.sum(centered_results < -abs(original_mean_diff),axis=2)
        returned_distribution = centered_results
    elif alternative == "greater":
        #are results greater than the original?
        p_val = np.sum(centered_results - (original_mean_diff) > 0,axis=2)
        returned_distribution = centered_results - abs(original_mean_diff)
    elif alternative == "less":
        #are results less than the original?
        p_val = np.sum(centered_results + (original_mean_diff) > 0,axis=2)
        returned_distribution = centered_results + abs(original_mean_diff)
    else:
        raise ValueError("alternative must be \"two-sided\", \"greater\", or \"less\"")
        
    return p_val / M, returned_distribution