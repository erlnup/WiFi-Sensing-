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

def hampel_filter(x, k=3, t0=3):
    n = len(x)
    y = x.copy()
    x_padded = np.pad(x, k, mode='reflect')
    
    def mad(arr):
        return np.median(np.abs(arr - np.median(arr)))
    
    for i in range(n):
        window = x_padded[i:i + 2 * k + 1]
        median_val = np.median(window)
        mad_val = mad(window)
        
        if np.abs(x[i] - median_val) > t0 * mad_val:
            y[i] = median_val
            
    return y

def filter_csi_matrix(csi_matrix: np.ndarray):
    filtered_rows = [hampel_filter(row) for row in csi_matrix]
    return np.array(filtered_rows)

class CSIAMPDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        mat = sio.loadmat(file_path)
        csi_amps = mat['CSIamp']
        
        filtered_array = filter_csi_matrix(csi_amps)
        tensor = torch.from_numpy(filtered_array).float()
        
        tensor = tensor.T
        
        person_id = os.path.basename(os.path.dirname(file_path))
        label = int(person_id) - 1 

        return tensor, label

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_dim = hidden_size

    def forward(self, x):
        outputs, (h_n, c_n) = self.LSTM(x)
        return h_n[-1]

class Encoder(nn.Module):
    def __init__(self, input_size, num_hidden_layers=16, num_stacked_lstm_encoders=3, encoder_type="lstm"):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.num_stacked_lstm_encoders = num_stacked_lstm_encoders
        self.output_dim = num_hidden_layers

        if self.encoder_type == "lstm":
            self.encoder = LSTMEncoder(self.input_size, num_hidden_layers, num_stacked_lstm_encoders)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, batch):
        output = self.encoder(batch)
        return output

class SignatureModel(nn.Module):
    def __init__(self, input_size, signature_dimensionality=10):
        super().__init__()
        self.linear = nn.Linear(input_size, signature_dimensionality)
    
    def forward(self, x):
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def train_model():
    folders = ['001', '002', '003']
    data_path_template = "/Users/erlingnupen/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/train_amp/{}/*.mat"

    all_files = []
    for folder in folders:
        all_files.extend(glob.glob(data_path_template.format(folder)))

    print(f"All files collected: {len(all_files)}")
    
    files_by_person = {}
    for file_path in all_files:
        person_id = os.path.basename(os.path.dirname(file_path))
        if person_id not in files_by_person:
            files_by_person[person_id] = []
        files_by_person[person_id].append(file_path)
    
    query_files = []
    gallery_files = []
    
    for person_id in sorted(files_by_person.keys()):
        files = sorted(files_by_person[person_id])
        split_point = len(files) // 2
        query_files.extend(files[:split_point])
        gallery_files.extend(files[split_point:split_point*2])

    print(f"Number of query files: {len(query_files)}")
    print(f"Number of gallery files: {len(gallery_files)}")

    query_dataset = CSIAMPDataset(query_files)
    gallery_dataset = CSIAMPDataset(gallery_files)
    
    batch_size = 38
    query_loader = DataLoader(dataset=query_dataset, batch_size=batch_size, shuffle=False)
    gallery_loader = DataLoader(dataset=gallery_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder_model = Encoder(input_size=342, encoder_type="lstm").to(device)
    signature_model = SignatureModel(input_size=encoder_model.output_dim).to(device)
    optimizer = Adam(list(encoder_model.parameters()) + list(signature_model.parameters()), lr=1e-3)
    
    loss_fn = nn.CrossEntropyLoss()
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        encoder_model.train()
        signature_model.train()
        
        total_loss = 0
        
        # Use zip to properly iterate over query and gallery batches
        for batch_idx, ((q_data, q_label), (g_data, g_label)) in enumerate(zip(query_loader, gallery_loader)):
            print(f"\nBatch {batch_idx + 1}")
            print(f"q_data original shape: {q_data.shape}, g_data original shape: {g_data.shape}")
            
            q_data, q_label = q_data.to(device), q_label.to(device)
            g_data, g_label = g_data.to(device), g_label.to(device)

            # Dynamic reshape: batch_size x timesteps x features
            q_data = q_data.view(q_data.size(0), -1, q_data.size(-1))
            g_data = g_data.view(g_data.size(0), -1, g_data.size(-1))
            print(f"q_data reshaped: {q_data.shape}, g_data reshaped: {g_data.shape}")

            encoded_queries = encoder_model(q_data)
            encoded_galleries = encoder_model(g_data)
            print(f"encoded_queries shape: {encoded_queries.shape}, encoded_galleries shape: {encoded_galleries.shape}")

            signature_queries = signature_model(encoded_queries)
            signature_galleries = signature_model(encoded_galleries)
            print(f"signature_queries shape: {signature_queries.shape}, signature_galleries shape: {signature_galleries.shape}")

            # Similarity matrix
            similarity_matrix = signature_queries @ signature_galleries.t()
            print(f"similarity_matrix shape: {similarity_matrix.shape}")
            
            # Target for cross-entropy
            target = torch.arange(similarity_matrix.size(0), device=device)
            loss = loss_fn(similarity_matrix, target)
            print(f"loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(query_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
        
    print("\n--- Training Complete. Saving models. ---")
    torch.save(encoder_model.state_dict(), 'encoder_model.pth')
    torch.save(signature_model.state_dict(), 'signature_model.pth')
    
    return encoder_model, signature_model


if __name__ == '__main__':
    train_model()
