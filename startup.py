import os
import argparse
import sys
import configparser
import ImageAnalysisModel as pa
import requests
import re

# def download_model(checkpoint_folder, file_url, file_name):
#     """
#     Check sam model exists or not ,if not downloaded
#     """
#     if not os.path.exists(checkpoint_folder):
#         os.makedirs(checkpoint_folder, exist_ok=True)
#         print(f"Checkpoints folder created: {checkpoint_folder}")
#
#     file_path = os.path.join(checkpoint_folder, file_name)
#     if not os.path.exists(file_path):
#         try:
#             print(f"Downloading model file: {file_name}...")
#             response = requests.get(file_url, stream=True)
#             response.raise_for_status()  # check whether the request is successful or not
#             with open(file_path, 'wb') as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     if chunk:
#                         f.write(chunk)
#             print("Sam model downloaded successfully")
#         except Exception as e:
#             print(f"Error occurred during sam model downloading : {e}")
#             sys.exit(1)
#     else:
#         print("Checkpoint already existed ，skip downloading")
# def parse_bins(industry_bins_string):
#             # Remove non-numeric characters and split by commas
#
#             # Assuming industry_bins_string looks something like "[38, 106, 1000, 8000]"
#             # Removing brackets and spaces before splitting
#             cleaned_string = re.sub(r'[\[\] ]', '', industry_bins_string)
#             bins_list = cleaned_string.split(',')
#
#             # Convert each element to an integer
#             try:
#                 industry_bins = [int(x) for x in bins_list]
#                 print("Parsed industry bins:", industry_bins)
#                 return industry_bins
#             except ValueError as e:
#                 print("Error converting to integer:", e)
#                 return []
def main(image_folder_path, containerWidth,config_path):
    """
    Main function to execute the image analysis process.
    """
    analyser = pa.ImageAnalysisModel(image_folder_path, containerWidth=containerWidth, config_path=config_path)
    analyser.run_analysis()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run particle analysis on the provided image folder.")
    parser.add_argument('image_folder_path', type=str, help='Path to the folder containing sample images.')
    parser.add_argument('--containerWidth', default=180000,type=int,help='Container width in micrometers, for example, 18 cm is equal to 180000 µm.')
    parser.add_argument('--config_path', type=str, default='config.ini',
                        help='config file path')
    args = parser.parse_args()

    main(args.image_folder_path,args.containerWidth,args.config_path)


