import os
import argparse
import sys

import ImageAnalysisModel as pa
import requests

def download_model(checkpoint_folder, file_url, file_name):
    """
    Check sam model exists or not ,if not downloaded
    """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)
        print(f"Checkpoints folder created: {checkpoint_folder}")

    file_path = os.path.join(checkpoint_folder, file_name)
    if not os.path.exists(file_path):
        try:
            print(f"Downloading model file: {file_name}...")
            response = requests.get(file_url, stream=True)
            response.raise_for_status()  # check whether the request is successful or not
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Sam model downloaded successfully")
        except Exception as e:
            print(f"Error occurred during sam model downloading : {e}")
            sys.exit(1)
    else:
        print("Checkpoint already existed ，skip downloading")

def main(image_folder_path, containerWidth):
    """
    Main function to execute the image analysis process.
    """
    if not image_folder_path or not containerWidth:
        print("Please provide image folder path as well as the container width.")
        return

    checkpoint_folder = 'checkpoints'
    model_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'
    model_name = 'sam2.1_hiera_large.pt'

    download_model(checkpoint_folder, model_url, model_name)

    analyser = pa.ImageAnalysisModel(image_folder_path, containerWidth)
    analyser.evenLighting()
    analyser.overlayImage()
    Testing = False
    analyser.analyseParticles(checkpoint_folder, Testing)
    analyser.saveSegments()
    industry_standard_bins = [38, 106, 1000, 8000]  # bins
    normal_bins = [ 1000, 2000,3000,4000,5000,6000,7000,8000,9000,10000]
    analyser.saveResults(industry_standard_bins)
    analyser.formatResults()
    analyser.saveResultsForNormalBinsOnly(normal_bins)
    analyser.formatResultsForNormalDistribution(True)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run particle analysis on the provided image folder.")
    parser.add_argument('image_folder_path', type=str, help='Path to the folder containing sample images.')
    parser.add_argument('containerWidth', type=int, help='Container width in micrometers, for example, 18 cm is equal to 180000 µm.')

    args = parser.parse_args()

    main(args.image_folder_path, args.containerWidth)
