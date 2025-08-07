
#from scipy.stats import median_absolute_deviation
import scipy
import numpy as np
import asyncio

async def hampel_filter(subcarrier: np.ndarray, window_size: int = 5):
    filtered = subcarrier.copy()
    n = len(subcarrier)
    k = window_size // 2

    for i in range(k, n - k):
        window = subcarrier[i - k : i + k + 1]
        median = np.median(window)
        mad = scipy.stats.median_abs_deviation(window)
        threshold = 3 * mad

        if np.abs(subcarrier[i] - median) > threshold:
            filtered[i] = median

    return filtered
        

    
    


    
