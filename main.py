import asyncio
import glob
import os
from numpy._core.multiarray import add_docstring 
import scipy.io as sio
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import time
from utils.hampel_filter import hampel_filter
import torch

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

files = glob.glob("/Users/erlingnupen/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/001/a*.mat")

mat = sio.loadmat('/Users/erlingnupen/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/001/a1.mat')

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

async def add_gaussian_noise(entry, sigma: float = 0.02):
    return entry + np.random.normal(loc=0.0, scale=sigma)

async def scale(entry):
    return entry * np.random.uniform(low=0.9, high=1.1)


async def time_shift(subcarrier: np.ndarray, int_pos, chunk_size: int = 100, shift_range: int = 5):
    """
    Applies time shift augmentation to a window (chunk) of the subcarrier array.
    Pads with mean if window is at the end.
    """
    chunk = subcarrier[int_pos:int_pos + chunk_size]

    if len(chunk) < chunk_size:
        # Handle empty chunk case for mean calculation
        mean_val = np.mean(chunk) if len(chunk) > 0 else 0
        try:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), constant_values=mean_val)
        except Exception as e:
            print(f"Padding failed in time shift augmentation, index: {int_pos} Exception: {e}")
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), constant_values=0)

    t_prime = np.random.randint(-shift_range, shift_range + 1)
    mean_val = np.mean(chunk)
    shifted_chunk = np.full_like(chunk, fill_value=mean_val)

    if t_prime > 0:
        shifted_chunk[t_prime:] = chunk[:-t_prime]
    elif t_prime < 0:
        shifted_chunk[:t_prime] = chunk[-t_prime:]
    else:
        shifted_chunk = chunk.copy()

    augmented_subcarrier = subcarrier.copy()
    augmented_subcarrier[int_pos:int_pos + chunk_size] = shifted_chunk

    

    return augmented_subcarrier

async def apply_random_augmentation(csi_matrix):

    augmentations = [add_gaussian_noise, scale, time_shift]
    augmented_matrix = csi_matrix.copy()
    n_rows, n_cols = augmented_matrix.shape

    for i in range(n_rows):
        j = 0
        while j < n_cols:
            if np.random.rand() < 0.9:
                augment = np.random.choice(augmentations)
                if augment is time_shift:
                    augmented_matrix[i, :] = await augment(augmented_matrix[i, :], j)
                    break
                else:
                    augmented_matrix[i, j] = await augment(augmented_matrix[i, j])
            j += 1

    return augmented_matrix






async def plot_augmented_matrix(csi_matrix, person):

    t_0 = time.time()

    filtered_array = await filter_csi_matrix(csi_matrix)
    augmented_array = await apply_random_augmentation(filtered_array)
    t_1 = time.time()

    t_diff = t_1 - t_0

    print(f"Filtering the array took: {t_diff} seconds")
    
    os.makedirs('visualizations', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(csi_matrix,
                cmap='viridis',
                cbar_kws={'label': '|H| amplitude'},
                xticklabels=False,
                yticklabels=False,
                ax=ax1)
    ax1.set_xlabel('Packet index')
    ax1.set_ylabel('Link × Sub-carrier index')
    ax1.set_title('Original CSI Amplitude Heat-map')

    sns.heatmap(augmented_array,
                cmap='viridis',
                cbar_kws={'label': '|H| amplitude'},
                xticklabels=False,
                yticklabels=False,
                ax=ax2)
    ax2.set_xlabel('Packet index')
    ax2.set_ylabel('Link × Sub-carrier index')
    ax2.set_title('Augmented CSI Amplitude Heat-map')

    plt.tight_layout()

    plt.savefig(f'visualizations/csi_comparison_{person}.png', dpi=300, bbox_inches='tight')
    #plt.savefig('visualizations/csi_comparison.pdf', bbox_inches='tight')  # Optional: also save as PDF
    plt.close()  # Close the figure to free memory

    print("Figure saved to visualizations/csi_comparison.png")


a_files = glob.glob("/Users/erlingnupen/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/001/a*.mat")

"""
for a_file in a_files:

    mat = sio.loadmat(a_file)

    print(f"mat file keys: {mat.keys()}")
    csi_amps = mat['CSIamp']

    person = os.path.basename(a_file)
"""


async def prepare_data():

    a_file = a_files[0]
    data = sio.loadmat(a_file)
    print(f"Loading file")

    csi_amps = data['CSIamp']

    print(f"File loaded")

    t_0 = time.time()
    print(f"Filtering array")
    filtered_array = await filter_csi_matrix(csi_amps)
    augmented_array = await apply_random_augmentation(filtered_array)
    t_1 = time.time()

    t_diff = t_1 - t_0
    print(f"Array filtered, augmentation took: {t_diff} seconds")

    matrix = augmented_array.reshape(342, 2000, 1)
    matrix = torch.from_numpy(matrix).float()

    return matrix, matrix.shape[2]





