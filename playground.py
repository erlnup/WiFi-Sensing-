import glob
import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F

class CSIAMPDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        mat = sio.loadmat(file_path)
        csi_amps = mat['CSIamp']
        
        tensor = torch.from_numpy(csi_amps).float()
        
        tensor = tensor.T
        
        person_id = os.path.basename(os.path.dirname(file_path))
        label = int(person_id) - 1 

        return tensor, label


data_path = "/Users/erlingnupen/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/train_amp/*/*.mat"
all_files = glob.glob(data_path)

files_by_person = {}
for file_path in all_files:
    person_id = os.path.basename(os.path.dirname(file_path))
    print(f"Person id: {person_id}")
    if person_id not in files_by_person:
        files_by_person[person_id] = []
    files_by_person[person_id].append(file_path)

query_files = []
gallery_files = []

for person_id in sorted(files_by_person.keys()):
    files = sorted(files_by_person[person_id])
    mid_point = len(files) // 2
    query_files.extend(files[:mid_point])
    gallery_files.extend(files[mid_point:])

print(f"Number of query files: {len(query_files)}")
print(f"Number of gallery files: {len(gallery_files)}")

query_dataset = CSIAMPDataset(query_files)
gallery_dataset = CSIAMPDataset(gallery_files)

batch_size = 1
query_loader = DataLoader(dataset=query_dataset, batch_size=batch_size, shuffle=False)
gallery_loader = DataLoader(dataset=gallery_dataset, batch_size=batch_size, shuffle=False)

for (q_data, q_label), (g_data, g_label) in zip(query_loader, gallery_loader):

    print(f"Query type: {type(q_label)}, Gallery data label: {type(g_label)}")
    print(f"Query : {q_label}, Gallery data label: {g_label}")
    print(f"Query data label: {q_label.item()}, Gallery data label: {g_label.item()}")

