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
from torch.utils.data import Sampler,BatchSampler,Dataset,DataLoader,TensorDataset
from tqdm import tqdm
from encoder_block import Encoder
from signature_model import SignatureModel
import torch.nn.functional as F
from torch.optim import Adam

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
csi_amps = mat['CSIamp']

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



async def prepare_data(a_file):

    data = sio.loadmat(a_file)
    print(f"Loading file")

    csi_amps = data['CSIamp']

    print(f"File loaded")

    t_0 = time.time()
    print(f"Filtering array")
    filtered_array = await filter_csi_matrix(csi_amps)
    print(f"Filtered Array shape: {np.shape(filtered_array)}")
    augmented_array = await apply_random_augmentation(filtered_array)
    t_1 = time.time()

    t_diff = t_1 - t_0
    print(f"Array filtered, augmentation took: {t_diff} seconds")

    matrix = augmented_array.reshape(342, 2000, 1)
    matrix = torch.from_numpy(matrix).float()

    return matrix, matrix.shape[2]


async def prepare_data_no_aug(a_file):
    print("Loading file...")
    data = sio.loadmat(a_file)

    csi_amps = data['CSIamp']  # Assuming shape fits your needs
    print("File loaded.")

    # Convert to torch tensor directly
    torch_tensor = torch.tensor(csi_amps, dtype=torch.float32)

    # Reshape if you know the exact expected shape
    matrix = torch_tensor.reshape(-1, 2000, 1)  # use -1 to infer first dim safely

    return matrix, matrix.shape[2]








async def custom_batch_selector(sample):
    return list(BatchSampler(sample))




async def bq_bg_samples(list_of_tensors):


    print("Creating tensor dataset")
    tensor_dataset = TensorDataset(*list_of_tensors)

    batch_size = 342
    print(f"Creating DataLoader with batch_size = {batch_size}")
    load_data = DataLoader(dataset=tensor_dataset, batch_size=batch_size)


    number_of_batches = 0

    bq_list = []
    bg_list = []

    for batch in tqdm(load_data):
        print(f"Length of batch: {len(batch)}")
        bq_list.append(batch[0])
        bg_list.append(batch[0])
        number_of_batches +=1

    print(f"Number of batches created from input matrix: {number_of_batches}")


    return (bq_list, bg_list)



async def prepare_full_session(a_files):
    # Split files: first half for query, second half for gallery
    mid_point = len(a_files) // 2
    query_files = a_files[:mid_point]
    gallery_files = a_files[mid_point:mid_point*2]  # Same people, different sessions
    
    query_tensors = []
    gallery_tensors = []
    
    for qfile, gfile in zip(query_files, gallery_files):
        #q_matrix, _ = await prepare_data(qfile)
        #g_matrix, _ = await prepare_data(gfile) 
        q_matrix, _ = await prepare_data_no_aug(qfile)
        g_matrix, _ = await prepare_data_no_aug(gfile)
        query_tensors.append(q_matrix)
        gallery_tensors.append(g_matrix)
    
    return query_tensors, gallery_tensors





async def DNN():

    a_files = glob.glob("/Users/erlingnupen/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/001/a*.mat")[0:10]

    print(f"Number of files loaded: {len(a_files)}")

    query_list, gallery_list = await prepare_full_session(a_files)

    #query_list = [tensor 1, tensor 2, trensor 3 ....]
    #gallery_list = [tensor 1, tensor 2, trensor 3 ....]


    #query_list, gallery_list = await bq_bg_samples(matrix)

    similarity_matrices = []

    
    
    encoder_model = Encoder(input_size=query_list[0].shape[2], encoder_type="lstm")
    signature_model = SignatureModel(input_size=encoder_model.output_dim)  # output_dim = embedding size
    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(signature_model.parameters()), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    

    device = torch.device("cpu")


       


    num_epochs = 10 # Example number of epochs
    for epoch in range(num_epochs):

        query_batch = torch.stack(query_list).to(device)      # Shape: [N, H, W] 
        gallery_batch = torch.stack(gallery_list).to(device)  # Shape: [M, H, W]
        
        encoder_model.train()
        signature_model.train()

        print(f"Query batch shape: {query_batch.shape}")
        print(f"galler batch shape: {gallery_batch.shape}") 

        #query_batch = query_batch.squeeze(-1)    # [19, 18, 2000, 1] → [19, 18, 2000]
        #gallery_batch = gallery_batch.squeeze(-1) # [19, 18, 2000, 1] → [19, 18, 2000]
        
        fin_query_batch = query_batch.view(-1, 2000, 1)    # 5 * 342 = 1710 samples
        fin_gallery_batch = gallery_batch.view(-1, 2000, 1)
        print(f"Query batch shape after squeeze: {fin_query_batch.shape}")
        print(f"galler batch shape after squeeze: {fin_gallery_batch.shape}")

        # Check this:
        print(f"Query and gallery are identical: {torch.equal(query_batch, gallery_batch)}")

        encoded_queries = encoder_model(fin_query_batch)        # Shape: [N, features]
        encoded_galleries = encoder_model(fin_gallery_batch)    # Shape: [M, features]
        
        signature_queries = signature_model(encoded_queries)   # Shape: [N, sig_features]  
        signature_galleries = signature_model(encoded_galleries) # Shape: [M, sig_features]
        
        similarity_matrix = signature_queries @ signature_galleries.t()  # Shape: [N, M]

        print(similarity_matrix)
        
        # result = torch.einsum('ij,kj->ik', signature_queries, signature_galleries)
        
        # Compute loss (you'll need to adjust target based on your loss function)
        target = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
        loss = loss_fn(similarity_matrix, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")





    # --- Save the trained models after the loop ---
    print("\n--- Training Complete. Saving models. ---")
    torch.save(encoder_model.state_dict(), 'encoder_model.pth')
    torch.save(signature_model.state_dict(), 'signature_model.pth')
    
    return encoder_model, signature_model




asyncio.run(DNN())



async def predict(trained_encoder, trained_signature_model):
    """
    This function takes the trained models and predicts the identity of an unseen query signal.
    """
    print("\n--- Starting Prediction on Unseen Data ---")

    # 1. Set models to evaluation mode
    trained_encoder.eval()
    trained_signature_model.eval()

    # 2. Prepare the Gallery (your database of known people)
    # Let's assume subjects 2, 3, and 4 are our known individuals in the gallery
    gallery_subjects = ['002', '003', '004']
    gallery_tensors = []
    print(f"Building gallery from subjects: {gallery_subjects}")
    for subject_id in gallery_subjects:
        # Load all files for a subject to represent them in the gallery
        gallery_files = glob.glob(f"/Users/erlingnupen/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/{subject_id}/a*.mat")
        gallery_matrix, _ = await prepare_full_session(gallery_files) # Process all their data
        gallery_tensors.append(gallery_matrix[0]) # Using first sample as representative

    # 3. Prepare the Unseen Query Data
    # Let's pretend this file is from an unknown person, but it's actually subject '003'
    unseen_query_files = glob.glob("/Users/erlingnupen/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/003/b1.mat")
    print(f"Loading unseen query sample: {unseen_query_files[0]}")
    unseen_query_tensor, _ = await prepare_full_session(unseen_query_files)

    # 4. Generate Signatures for Gallery and Query
    # Use torch.no_grad() to disable gradient calculations, saving memory and computation
    with torch.no_grad():
        # Create signatures for each person in the gallery
        gallery_signatures = []
        for gallery_person_tensor in gallery_tensors:
             # Important: The models are now the trained ones
            encoded_gallery = trained_encoder(gallery_person_tensor)
            signature_gallery = trained_signature_model(encoded_gallery)
            # Average the signatures if a person has multiple samples
            gallery_signatures.append(signature_gallery.mean(dim=0)) 
        
        gallery_signatures = torch.stack(gallery_signatures)

        # Create the signature for the single unseen query
        encoded_query = trained_encoder(unseen_query_tensor[0])
        query_signature = trained_signature_model(encoded_query)


    # 5. Compare and Find the Best Match
    # Calculate similarity between the query and everyone in the gallery
    similarity_scores = query_signature @ gallery_signatures.t()

    # Convert scores to probabilities
    probabilities = F.softmax(similarity_scores, dim=1)
    
    # Get the top prediction
    top_prob, top_idx = torch.max(probabilities, dim=1)
    
    for idx in top_idx:
        predicted_subject_id = gallery_subjects[idx.item()]
        print(predicted_subject_id)
    
    print("\n--- Prediction Result ---")
    print(f"Probabilities of matching each gallery subject: {probabilities.numpy()[0]}")
    print(f"The model predicts the person is: Subject {predicted_subject_id}")
    print(f"Confidence: {top_prob.item() * 100:.2f}%")


async def main():
    # First, train the model and get the trained instances
    encoder, signature = await DNN()
    
    # Now, use the trained models to make a prediction
    await predict(encoder, signature)

# Run the main asynchronous function
asyncio.run(main())





    





    
