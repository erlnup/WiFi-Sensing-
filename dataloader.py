from torch.utils.data import Dataset, DataLoader
import scipy.io as sio

class CSIAMPDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform # Optional data augmentation

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        mat = sio.loadmat(file_path)
        csi_amp = mat['CSIamp']
        csi_tensor = torch.from_numpy(csi_amp).float()
        
        # Add a channel dimension for consistency (342, 2000) -> (342, 2000, 1)
        csi_tensor = csi_tensor.unsqueeze(-1)

        # Apply transformations if they exist
        if self.transform:
            csi_tensor = self.transform(csi_tensor)

        # The label is the person ID, which you can extract from the filename path
        # e.g., '.../test_amp/001/a1.mat' -> '001'
        person_id = os.path.basename(os.path.dirname(file_path))
        label = int(person_id)

        return csi_tensor, label
