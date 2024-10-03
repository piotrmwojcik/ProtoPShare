import os
import glob

# Define the root directory where you want to search
root_dir = "/data/pwojcik/ProtoPShare/saved_models/resnet18/003/16push0.8675_nearest_train/"

# Use glob to search for files matching the pattern in all subdirectories
pattern = os.path.join(root_dir, '**', 'nearest-*_original_hp_mask.png')

# Use glob with recursive search
files = glob.glob(pattern, recursive=True)

# Print the list of found files
for file_path in files:
    print(file_path)