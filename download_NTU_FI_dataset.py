import kagglehub

# Download latest version
path = kagglehub.dataset_download("hylanj/wifi-csi-dataset-ntu-fi-humanid")

print("Path to dataset files:", path)
