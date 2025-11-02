import kagglehub

# Download latest version
path = kagglehub.dataset_download("karagwaanntreasure/plant-disease-detection")

print("Path to dataset files:", path)
