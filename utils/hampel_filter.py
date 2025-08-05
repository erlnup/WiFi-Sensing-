
#from scipy.stats import median_absolute_deviation
import scipy
import numpy as np
import asyncio

async def hampel_filter(subcarrier: np.ndarray, window_size: int = 5):
    filtered = subcarrier.copy()
    n = len(subcarrier)

    for i in range(n - window_size + 1):
        window = filtered[i:i + window_size]
        median = np.median(window)
        mad = scipy.stats.median_abs_deviation(window)

        threshold = 3 * mad
        mask = np.abs(window - median) > threshold

        indices = np.arange(i, i + window_size)
        filtered[indices[mask]] = 0

    # Optionally, handle the last few elements (edges) here if needed

    return filtered
        

    
    


    
