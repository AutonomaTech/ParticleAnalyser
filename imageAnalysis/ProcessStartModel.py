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

    config.read(defaultConfigPath, encoding='utf-8')
    defaultContainerWidth = abs(int(config.get('analysis', 'containerWidth', fallback=180000)))
    defaultOutputfolder = config.get('analysis', 'defaultOutputfolder', fallback="defaultProgram")
    defaultScalingNumber = int(config.get('analysis', 'scalingNumber', fallback=0))
    # Define the SMB server and share
    FS_DIRECT_PATH = str(config.get('FileIn', 'DIRECT_PATH', fallback=""))
    
    defaultTemperature = int(config.get('analysis', 'temperature', fallback=3000))
    defaultOriTemperature = int(config.get('analysis', 'ori_temperature', fallback=2500))

except Exception as e:
    print(f"Unexpected error: {e}")
    defaultContainerWidth = 180000
    defaultOutputfolder = "defaultProgram"
    defaultScalingNumber = 0

# Logger setup
logger = get_logger("StartUp")

class ProcessStartModel:

    def __init__(self, picturePath=None, jsonPath=None,sampleID=None, programNumber=None, weight=None, timestamp=None, checkpoint_folder=None, SAMPLEFOLDER=None):
        self.SAMPLEFOLDER=SAMPLEFOLDER
        # Extract sample ID and timestamp
        base_sample_id, extracted_timestamp, sampleID = self.extract_sample_id_and_timestamp(sampleID)
        timestamp = timestamp or extracted_timestamp



        #  create complete file path for the result folder
        # result_folder_path = os.path.join(self.SAMPLEFOLDER,  sampleID)
        # os.makedirs( result_folder_path, exist_ok=True)
        #
        # # Original file path (in SMB)
        # source_bmp_path = os.path.join(FS_DIRECT_PATH, f"{sampleID}.bmp")
        # source_json_path = os.path.join(FS_DIRECT_PATH, f"{sampleID}.json")

        #  Target file path (in the result folder)
        # target_bmp_path = os.path.join( result_folder_path, f"{sampleID}.bmp")
        # target_json_path = os.path.join( result_folder_path, f"{sampleID}.json")
        #
        # #  check if original file existed
        # if not os.path.exists(source_json_path):
        #     logger.error(f"JSON File does not exist: {source_json_path}")
        #     raise FileNotFoundError(f"JSON file not found: {source_json_path}")
        # if not os.path.exists(source_bmp_path):
        #     logger.error(f"BMP File does not exist: {source_bmp_path}")
        #     raise FileNotFoundError(f"BMP file not found: {source_bmp_path}")
        #
        # # move file to pre-created result folder
        # try:
        #     shutil.move(source_bmp_path, target_bmp_path)
        #     logger.info(f"Moved {sampleID}.bmp to { result_folder_path}")
        #
        #     shutil.move(source_json_path, target_json_path)
        #     logger.info(f"Moved {sampleID}.json to { result_folder_path}")
        #
        # except Exception as e:
        #     logger.error(f"Error moving files to result folder: {str(e)}")
        #     raise
        #
        # self.bmp_file_path = target_bmp_path
        # self.json_file_path = target_json_path
        # Process JSON file
        try:
            with open(jsonPath, 'r') as f:
                json_data = json.load(f)

            if 'Temperature' not in json_data:
                logger.info("Warning: 'Temperature' key is missing. Using defaultTemperature.")
            temperature = defaultTemperature if json_data.get('Temperature', 0) <= 0 else json_data['Temperature']

            if 'OriTemperature' not in json_data:
                logger.info("Warning: 'OriTemperature' key is missing. Using defaultOriTemperature.")
            ori_temperature = defaultOriTemperature if json_data.get('OriTemperature', 0) <= 0 else json_data['OriTemperature']


            json_data['programId'] = programNumber
            json_data['sampleID'] = base_sample_id if sampleID else None
            json_data['timestamp'] = timestamp if timestamp else None
            json_data['weight'] = weight if weight else None



            # Add CustomField attributes dynamically
            custom_fields = {key: value for key, value in json_data.items() if key.startswith('CustomField')}
            for idx, (key, value) in enumerate(custom_fields.items(), start=1):
                setattr(self, f"CustomField{idx}", value)
            
            # Extract crop coordinates if they exist
            _crop_coords = {}
            crop_keys = [
                "CropTopLeftX", "CropTopLeftY",
                "CropTopRightX", "CropTopRightY",
                "CropBottomRightX", "CropBottomRightY",
                "CropBottomLeftX", "CropBottomLeftY"
            ]
            for key in crop_keys:
                if key in json_data:
                    _crop_coords[key] = json_data[key]

            with open(jsonPath, 'w') as f:
                json.dump(json_data, f, indent=4)

        except FileNotFoundError:
            logger.error(f"File {sampleID} or its JSON counterpart not found at {picturePath}")
            raise
        except Exception as e:
            logger.error(f"Error while updating JSON file for {sampleID}: {str(e)}")
            raise

        # Update instance variables

        self.picturePath = picturePath
        self.sampleID = sampleID
        self.programNumber = programNumber
        self.weight = weight
        self.ori_temperature=ori_temperature
        self.temperature=temperature
        self.timestamp = timestamp
        self.checkpoint_folder = checkpoint_folder
        self.crop_coords = _crop_coords

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
                ori_temperature=self.ori_temperature,
                temperature=self.temperature,
                sampleID=self.sampleID,
                config_path=config_path,
                crop_coords=self.crop_coords,
                **custom_fields
            )
            analyser.run_analysis(testing)
        except Exception as e:
            logger.error(f"Fatal error in main execution: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise



