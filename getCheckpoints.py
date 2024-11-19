# Downloads the SAM2 checkpoints from the provided URL and saves them locally.

import os
import requests

# URL of the checkpoint file
url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

# Path to save the file
# Update this path if necessary
folder_path = "C:/Users/marco/Desktop/particle_analyser/checkpoints"
file_path = os.path.join(folder_path, "sam2.1_hiera_large.pt")

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Download the file
response = requests.get(url)
with open(file_path, "wb") as file:
    file.write(response.content)

print(f"File downloaded to {file_path}")
