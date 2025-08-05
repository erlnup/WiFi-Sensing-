import asyncio
import glob

from numpy._core.multiarray import add_docstring 
import scipy.io as sio
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from utils.hampel_filter import hampel_filter
NTU_FI = '.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID'


"""
NTU_Fi_HumanID  test_amp  train_amp

NTU_Fi_HumanID:
 test_amp   train_amp


train_amp:

 001   002   003   004   005   006   007   008   009   010   011   012   013   015

test_amp:
 001   002   003   004   005   006   007   008   009   010   011   012   013   015

 001:
 a0.mat    a11.mat   a3.mat   a6.mat   a9.mat   b10.mat   b2.mat   b5.mat   b8.mat   c1.mat    c12.mat   c4.mat   c7.mat
 a1.mat    a12.mat   a4.mat   a7.mat   b0.mat   b11.mat   b3.mat   b6.mat   b9.mat   c10.mat   c2.mat    c5.mat   c8.mat
 a10.mat   a2.mat    a5.mat   a8.mat   b1.mat   b12.mat   b4.mat   b7.mat   c0.mat   c11.mat   c3.mat    c6.mat   c9.mat

"""

full_path = '/Users/erlnup/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/001/a0.mat'

files = glob.glob("/Users/erlnup/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/001/a*.mat")

mat = sio.loadmat('/Users/erlnup/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/001/a1.mat')

# Person ID: 001, session a:
print(f"mat file keys: {mat.keys()}")
csi_amps = mat['CSIamp']

print(f' Type {type(csi_amps)}')
print(f' Shape CSIamp: {np.shape(csi_amps)}')
print(f' First element {csi_amps[0]}')
print(f'First element Type {type(csi_amps[0])}')
print(f'test {csi_amps[0][0]}')

filter_csi_amps = np.empty_like(csi_amps)

async def filter_csi_matrix(csi_matrix: np.ndarray):
    # hampel_filter is async, it is therefore automatically a coroutine that gets gathered and run later with asyncio gather 
    tasks = [hampel_filter(row) for row in csi_matrix]
    filtered_rows = await asyncio.gather(*tasks)
    return np.array(filtered_rows)

async def add_gaussian_noise(csi_matrix, sigma: float = 0.02):
    noise = np.random.normal(loc=0.0, scale=sigma, size=csi_matrix.shape)
    
    mask = np.zeros(csi_matrix.size, dtype=bool)
    num_noisy = int(csi_matrix.size * 0.9)
    mask[:num_noisy] = True
    np.random.shuffle(mask)
    
    noise_mask = mask.reshape(csi_matrix.shape)
    
    noisy_matrix = csi_matrix.copy()
    noisy_matrix[noise_mask] += noise[noise_mask]
    
    return noisy_matrix

async def scale_amplitude(csi_matrix):
    scale_factors = np.random.uniform(low=0.9, high=1.1, size=csi_matrix.shape)
    return csi_matrix * scale_factors

async def time_shift(matrix: np.ndarray, chunk_size: int = 100, shift_range: int = 5):
    shifted_matrix = []

    for subcarrier in matrix:  # each row
        shifted_subcarrier = []

        for i in range(0, len(subcarrier), chunk_size):
            chunk = subcarrier[i:i + chunk_size]

            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), constant_values=np.mean(chunk))

            t_prime = np.random.randint(-shift_range, shift_range + 1)
            mean_val = np.mean(chunk)
            shifted_chunk = np.full_like(chunk, fill_value=mean_val)

            if t_prime > 0:
                shifted_chunk[t_prime:] = chunk[:-t_prime]
            elif t_prime < 0:
                shifted_chunk[:t_prime] = chunk[-t_prime:]
            else:
                shifted_chunk = chunk.copy()

            shifted_subcarrier.append(shifted_chunk)

        full_shifted = np.concatenate(shifted_subcarrier)
        shifted_matrix.append(full_shifted)

    return np.array(shifted_matrix)

async def apply_random_augmentation(csi_matrix):

#generate a random mask that where each entry has 90 % of being true:
mask = np.random.rand(*filtered_array.shape) < 0.9

# time shifting has to be done in batches


# MODIFIED: Combined plotting function for comparison
async def plot_augmented_matrix(csi_matrix):
    # Apply augmentations
    add_noise = await add_gaussian_noise(csi_matrix)
    scale_amps = await scale_amplitude(add_noise)
    time_shifted = await time_shift(scale_amps)
    filtered_array = await filter_csi_matrix(time_shifted)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Original data
    sns.heatmap(csi_matrix,
                cmap='viridis',
                cbar_kws={'label': '|H| amplitude'},
                xticklabels=False,
                yticklabels=False,
                ax=ax1)
    ax1.set_xlabel('Packet index')
    ax1.set_ylabel('Link × Sub-carrier index')
    ax1.set_title('Original CSI Amplitude Heat-map')
    
    # Right plot: Augmented data
    sns.heatmap(filtered_array,
                cmap='viridis',
                cbar_kws={'label': '|H| amplitude'},
                xticklabels=False,
                yticklabels=False,
                ax=ax2)
    ax2.set_xlabel('Packet index')
    ax2.set_ylabel('Link × Sub-carrier index')
    ax2.set_title('Augmented CSI Amplitude Heat-map')
    
    plt.tight_layout()
    plt.show()

# Run the comparison
asyncio.run(plot_augmented_matrix(csi_amps))




