import sys
import os
import json
import requests
import time
import configparser
import traceback
from logger_config import get_logger
sys.path.append(os.path.join(os.getcwd(), "imageAnalysis"))
ProcessStartModel = __import__("ProcessStartModel")

# Constants
BASEFOLDER = os.path.abspath(os.path.join(os.getcwd(), "CapturedImages"))
SAMPLEFOLDER = os.path.abspath(os.path.join(os.getcwd(), "Samples"))
defaultConfigPath = 'config.ini'
try:
    # Load configuration file
    config = configparser.ConfigParser()
    config.read(defaultConfigPath)

    # Read values from config.ini
    defaultContainerWidth = int(config.get(
        'analysis', 'containerWidth', fallback=180000))
    defaultOutputfolder = config.get(
        'analysis', 'defaultOutputfolder', fallback="defaultProgram")

except Exception as e:
    print(f"Unexpected error: {e}")
    defaultContainerWidth = 180000
    defaultOutputfolder = "defaultProgram"

#SAM2
checkpoint_folder = os.path.join(os.getcwd(), "sam2\\checkpoints")
model_name = 'sam2.1_hiera_large.pt'
model_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'


# Logger setup
logger = get_logger("StartUp")
def analyze_folder(folder_path):
    """ Continuously analyze files in the folder for BMP and corresponding JSON files. """
    while True:
        print("Monitoring folder...")
        for filename in os.listdir(folder_path):
            if filename.endswith('.bmp'):
                bmp_file = os.path.join(folder_path, filename)
                json_file = bmp_file.replace('.bmp', '.json')

                if os.path.exists(json_file):
                    try:
                        with open(json_file, 'r') as f:
                            json_data = json.load(f)

                        newImage = ProcessStartModel.ProcessStartModel(
                            picturePath=folder_path, sampleID=filename, programNumber=int(json_data.get('programNumber')), checkpoint_folder=checkpoint_folder)
                        logger.info(
                            f"Initialized ProcessStartModel for {bmp_file}")

                        newImage.analyse(testing=True)
                    except Exception as e:
                        logger.error(
                            f"Error processing {bmp_file} and {json_file}: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.error(f"Missing JSON file for {bmp_file}")

        time.sleep(1)

def get_remote_file_size(url):
    """Get the file size from the server (Content-Length)."""
    response = requests.head(url)
    if 'Content-Length' in response.headers:
        return int(response.headers['Content-Length'])
    return None  # Return None if size can't be determined

def download_model():
    """Download the model file if missing or incomplete."""
    try:
        os.makedirs(checkpoint_folder, exist_ok=True)

        remote_size = get_remote_file_size(model_url)
        if remote_size is None:
            logger.warning("Could not determine file size. Proceeding without verification.")
        file_path = os.path.join(checkpoint_folder, model_name)
        # Check if file exists and matches expected size
        if os.path.exists(file_path):
            local_size = os.path.getsize(file_path)
            if remote_size and local_size == remote_size:
                logger.info("Model already exists and is fully downloaded.")
                return
            else:
                logger.warning("Incomplete or corrupted file detected! Deleting and redownloading...")
                os.remove(file_path)

        logger.info(f"Downloading model {model_name}...")

        headers = {}
        if os.path.exists(file_path):
            existing_size = os.path.getsize(file_path)
            headers['Range'] = f'bytes={existing_size}-'  # Resume download

        response = requests.get(model_url, headers=headers, stream=True)
        response.raise_for_status()

        with open(file_path, 'ab') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Final file size check
        if remote_size and os.path.getsize(file_path) != remote_size:
            logger.error("Download incomplete. Retrying...")
            os.remove(file_path)
            download_model()  # Retry download

        logger.info("Download completed successfully!")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == '__main__':
    for folder in [BASEFOLDER, SAMPLEFOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder {folder} created.")
        else:
            print(f"Folder {folder} already exists.")
    download_model()
    analyze_folder(BASEFOLDER)