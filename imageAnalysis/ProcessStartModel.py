
import json
import configparser
import traceback
from logger_config import get_logger
import ImageAnalysisModel as pa
import os
import shutil 

defaultConfigPath = 'config.ini'

try:
    config = configparser.ConfigParser()
    config.read(defaultConfigPath)

    defaultContainerWidth = abs(int(config.get('analysis', 'containerWidth', fallback=180000)))
    defaultOutputfolder = config.get('analysis', 'defaultOutputfolder', fallback="defaultProgram")
    defaultScalingNumber = int(config.get('analysis', 'scalingNumber', fallback=0))
    # Define the SMB server and share
    SMB_SERVER = str(config.get('SMBServer', 'SMB_SERVER', fallback="AT-SERVER"))
    SMB_SHARE = str(config.get('SMBServer', 'SMB_SHARE', fallback="ImageDataShare"))

except Exception as e:
    print(f"Unexpected error: {e}")
    defaultContainerWidth = 180000
    defaultOutputfolder = "defaultProgram"
    defaultScalingNumber = 0

# Logger setup
logger = get_logger("StartUp")

class ProcessStartModel:

    def __init__(self, picturePath=None, sampleID=None, programNumber=None, weight=None, timestamp=None, checkpoint_folder=None, SAMPLEFOLDER=None):
        self.SAMPLEFOLDER=SAMPLEFOLDER
        # Extract sample ID and timestamp
        base_sample_id, extracted_timestamp, sampleID = self.extract_sample_id_and_timestamp(sampleID)
        timestamp = timestamp or extracted_timestamp

        # Create folder path
        folder_path = os.path.join(
            self.SAMPLEFOLDER, programNumber if programNumber not in [None, 0] else "defaultProgram", sampleID
        )
        os.makedirs(folder_path, exist_ok=True)

        new_json_name = f"{base_sample_id}.json"
        new_json_path = os.path.join(folder_path, new_json_name)

        # Move .bmp and .json files from SMB share
        for ext in ['.bmp', '.json']:
            remote_file_path = f"\\\\{SMB_SERVER}\\{SMB_SHARE}\\{sampleID}{ext}"
            new_file_path = os.path.join(folder_path, f"{base_sample_id}{ext}")
            try:
                print(remote_file_path)
                print(new_file_path)
                shutil.move(remote_file_path, new_file_path)
            except Exception as e:
                logger.error(f"Error while copying {sampleID}{ext} from SMB: {str(e)}")
                raise

        # Process JSON file
        try:
            with open(new_json_path, 'r') as f:
                json_data = json.load(f)

            json_data['programId'] = programNumber
            json_data['sampleID'] = base_sample_id if sampleID else None
            json_data['timestamp'] = timestamp if timestamp else None
            json_data['weight'] = weight if weight else None

            # Add CustomField attributes dynamically
            custom_fields = {key: value for key, value in json_data.items() if key.startswith('CustomField')}
            for idx, (key, value) in enumerate(custom_fields.items(), start=1):
                setattr(self, f"CustomField{idx}", value)

            with open(new_json_path, 'w') as f:
                json.dump(json_data, f, indent=4)

        except FileNotFoundError:
            logger.error(f"File {sampleID} or its JSON counterpart not found at {picturePath}")
            raise
        except Exception as e:
            logger.error(f"Error while updating JSON file for {sampleID}: {str(e)}")
            raise

        # Update instance variables
        self.picturePath = folder_path
        self.sampleID = base_sample_id
        self.programNumber = programNumber
        self.weight = weight
        self.timestamp = timestamp
        self.checkpoint_folder = checkpoint_folder

    def extract_sample_id_and_timestamp(self, sample_id):
        """ Extracts the base sample ID and timestamp from the filename. """
        if len(sample_id) > 15 and (sample_id.endswith('.bmp') or sample_id.endswith('.json')):
            no_extension_sample_id = sample_id.rsplit('.', 1)[0]
            base_sample_id = no_extension_sample_id[:-16]
            timestamp = no_extension_sample_id[-15:]
            return base_sample_id, timestamp, no_extension_sample_id
        return sample_id, None

        
    def analyse(self, container_width=defaultContainerWidth, defaultScalingNumber=defaultScalingNumber, config_path=defaultConfigPath, testing=False):
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
                scalingNumber=defaultScalingNumber,  
                containerWidth=container_width,
                checkpoint_folder=self.checkpoint_folder,
                sampleID=self.sampleID,
                config_path=config_path,
                **custom_fields
            )
            analyser.run_analysis(testing)
        except Exception as e:
            logger.error(f"Fatal error in main execution: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise



