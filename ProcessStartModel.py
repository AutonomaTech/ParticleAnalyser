
import sys
import os
import json
import shutil
import time
import configparser
import traceback
from logger_config import get_logger
print(os.path.join(os.getcwd(), "imageAnalysis"))
sys.path.append(os.path.join(os.getcwd(), "imageAnalysis"))
pa = __import__("ImageAnalysisModel")


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


# Logger setup
logger = get_logger("StartUp")


def extract_sample_id_and_timestamp(sample_id):
    """ Extracts the base sample ID and timestamp from the filename. """
    if len(sample_id) > 15 and (sample_id.endswith('.bmp') or sample_id.endswith('.json')):
        no_extension_sample_id = sample_id.rsplit('.', 1)[0]
        base_sample_id = no_extension_sample_id[:-16]
        timestamp = no_extension_sample_id[-15:]
        return base_sample_id, timestamp, no_extension_sample_id
    return sample_id, None


class ProcessStartModel:

    def __init__(self, picturePath=None, sampleID=None, programNumber=None, weight=None, CustomField1=None, CustomField2=None, timestamp=None):
        # Extract sample ID and timestamp
        base_sample_id, extracted_timestamp, sampleID = extract_sample_id_and_timestamp(
            sampleID)
        timestamp = timestamp or extracted_timestamp

        # Create folder path
        folder_path = os.path.join(
            SAMPLEFOLDER, programNumber if programNumber not in [
                None, 0] else "defaultProgram", sampleID
        )
        os.makedirs(folder_path, exist_ok=True)

        new_json_name = f"{base_sample_id}.json"
        new_json_path = os.path.join(folder_path, new_json_name)

        # Move .bmp and .json files to the new folder
        for ext in ['.bmp', '.json']:
            old_file_path = os.path.join(picturePath, f"{sampleID}{ext}")
            new_file_path = os.path.join(folder_path, f"{base_sample_id}{ext}")
            try:
                if os.path.exists(old_file_path):
                    shutil.move(old_file_path, new_file_path)
                else:
                    logger.error(
                        f"File {sampleID}{ext} not found at {picturePath}")
                    raise FileNotFoundError(f"File {sampleID}{ext} not found.")
            except Exception as e:
                logger.error(
                    f"Error while moving file {sampleID}{ext}: {str(e)}")
                raise

        # Update the JSON file with the extracted timestamp
        try:
            with open(new_json_path, 'r') as f:
                json_data = json.load(f)
            json_data['sampleID'] = base_sample_id  # Update sampleID
            json_data['timestamp'] = timestamp  # Add timestamp to JSON

            # Dynamically add CustomField attributes based on the JSON keys
            custom_fields = {key: value for key, value in json_data.items(
            ) if key.startswith('CustomField')}
            for idx, (key, value) in enumerate(custom_fields.items(), start=1):
                setattr(self, f"CustomField{idx}", value)

            with open(new_json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
        except FileNotFoundError:
            logger.error(
                f"File {sampleID} or its JSON counterpart not found at {picturePath}")
            raise
        except Exception as e:
            logger.error(
                f"Error while updating JSON file for {sampleID}: {str(e)}")
            raise

        # Update instance variables
        self.picturePath = folder_path
        self.sampleID = base_sample_id
        self.programNumber = programNumber
        self.weight = weight
        self.timestamp = timestamp

    def analyse(self, container_width=defaultContainerWidth, config_path=defaultConfigPath, testing=False):
        """ Main function to execute the image analysis process. """
        try:
            logger.info(
                f"Starting analysis with folder: {self.picturePath}, container width: {container_width}"
            )

            # Collect dynamically added custom fields
            custom_fields = {attr: getattr(self, attr) for attr in dir(
                self) if attr.startswith("CustomField")}

            # Pass all attributes, including dynamically added ones, to the model
            analyser = pa.ImageAnalysisModel(
                image_folder_path=self.picturePath,
                containerWidth=container_width,
                sampleID=self.sampleID,
                config_path=config_path,
                **custom_fields
            )
            analyser.run_analysis(testing)
        except Exception as e:
            logger.error(f"Fatal error in main execution: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


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

                        newImage = ProcessStartModel(
                            picturePath=folder_path, sampleID=filename, programNumber=int(json_data.get('programNumber')))
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


if __name__ == '__main__':
    for folder in [BASEFOLDER, SAMPLEFOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder {folder} created.")
        else:
            print(f"Folder {folder} already exists.")
    analyze_folder(BASEFOLDER)
