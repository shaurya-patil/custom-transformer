import kagglehub

# Download latest version
path = kagglehub.dataset_download("alincijov/bilingual-sentence-pairs")

print("Path to dataset files:", path)