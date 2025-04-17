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
    # Define the SMB server and share
    SMB_SERVER = str(config.get('SMBServer', 'SMB_SERVER', fallback="AT-SERVER"))
    SMB_SHARE = str(config.get('SMBServer', 'SMB_SHARE', fallback="ImageDataShare"))

except Exception as e:
    print(f"Unexpected error: {e}")
    defaultContainerWidth = 180000
    defaultOutputfolder = "defaultProgram"

#SAM2
checkpoint_folder = os.path.join(os.getcwd(), "sam2\\checkpoints")
model_name = 'sam2.1_hiera_large.pt'
model_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'

# SMBServer
BASEFOLDER = f"\\\\{SMB_SERVER}\\{SMB_SHARE}"
SAMPLEFOLDER = os.path.abspath(os.path.join(BASEFOLDER, "Samples"))


# Logger setup
logger = get_logger("StartUp")

def analyze_folder(folder_path):
    """ Continuously analyze files in the folder for BMP and corresponding JSON files. """
    while True:
        print("Monitoring folder...")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Ensure it's a file, not a directory
            if os.path.isfile(file_path):
                if filename.endswith('.bmp'):
                    bmp_file = file_path
                    json_file = bmp_file.replace('.bmp', '.json')

                    if os.path.exists(json_file):
                        try:
                            with open(json_file, 'r') as f:
                                json_data = json.load(f)

                            program_id = int(json_data.get("programId", 0))
                            # Assuming ProcessStartModel and logger are defined elsewhere
                            newImage = ProcessStartModel.ProcessStartModel(
                                picturePath=folder_path, 
                                sampleID=filename, 
                                programNumber=program_id, 
                                checkpoint_folder=checkpoint_folder, 
                                SAMPLEFOLDER=SAMPLEFOLDER
                            )
                            logger.info(f"Initialized ProcessStartModel for {bmp_file}")

                            newImage.analyse(testing=True)
                            updateStatusJson()
                        except Exception as e:
                            logger.error(f"Error processing {bmp_file} and {json_file}: {str(e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                    else:
                        logger.error(f"Missing JSON file for {bmp_file}")
                        continue

        time.sleep(1)
def updateStatusJson():
    machineStatusJson=os.path.abspath(os.path.join(BASEFOLDER, "machineStatus.json"))

    try:
        with open(machineStatusJson, 'r') as f:
            data = json.load(f)

        # Check if 'ProcessCount' key exists
        if 'ProcessCount' in data:
            # Decrease the ProcessCount by 1
            data['ProcessCount'] = max(0, data['ProcessCount'] - 1)

            print(f"Updated ProcessCount in {machineStatusJson}: {data['ProcessCount']}")

        else:
            data['ProcessCount'] = 0
            
        with open(machineStatusJson, 'w') as f:
            json.dump(data, f, indent=4)

    except Exception as e:
        print(f"Error updating status file {machineStatusJson}: {e}")

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

def check_remote_folder_available(remote_folder):
    """Check if the remote folder is accessible and openable."""
    try:
        # First, check if the folder exists
        if os.path.exists(remote_folder):
            # Try to list the contents of the folder to ensure it is accessible
            if os.path.isdir(remote_folder):
                # Try to open a file within the folder or list files
                try:
                    # Attempt to open a file or list files
                    files = os.listdir(remote_folder)  # Listing the contents verifies accessibility
                    return True
                except PermissionError:
                    print(f"PermissionError: Unable to access the folder {remote_folder}. Check permissions.")
                    return False
                except OSError as e:
                    print(f"OSError: Unable to access the folder {remote_folder}. Error: {e}")
                    return False
            else:
                print(f"The path {remote_folder} is not a directory.")
                return False
        else:
            print(f"Remote folder {remote_folder} does not exist.")
            return False
    except Exception as e:
        print(f"Error accessing remote folder {remote_folder}: {e}")
        return False

def create_remote_folder(remote_folder):
    """Checks if remote folder is available, retries until it is accessible."""
    print(f"Checking if remote folder {remote_folder} is available...")
    retries = 0
    while not check_remote_folder_available(remote_folder):  # Retry indefinitely
        retries += 1
        print(f"Remote folder {remote_folder} not available. Retrying... Attempt {retries}")
        time.sleep(5)  # Retry after 5 seconds
    print(f"Remote folder {remote_folder} is now available.")

if __name__ == '__main__':    
    create_remote_folder(BASEFOLDER)

    if not os.path.exists(SAMPLEFOLDER):
        os.makedirs(SAMPLEFOLDER)
        print(f"Local folder {SAMPLEFOLDER} created.")
    else:
        print(f"Local folder {SAMPLEFOLDER} already exists.")

    download_model()
    analyze_folder(BASEFOLDER)
