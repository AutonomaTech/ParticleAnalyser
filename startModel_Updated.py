import sys
import os
import json
import requests
import time
import configparser
import queue
from datetime import datetime
import pytz
from collections import defaultdict
import threading
import shutil
import subprocess
import inspect
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from logger_config import get_logger
# Logger setup
logger = get_logger("StartUp")
from Config_Manager import ConfigManager


sys.path.append(os.path.join(os.getcwd(), "imageAnalysis"))
sys.path.append(os.path.join(os.getcwd(), 'ImagePreprocessing'))

# fix inspect.getsource problem
original_getsource = inspect.getsource

def fixed_getsource(obj):
    try:
        return original_getsource(obj)
    except OSError:
        # return Enum._generate_next_value_ standard implementation
        return '''def _generate_next_value_(name, start, count, last_values):
    if not count:
        return start if start is not None else 1
    try:
        return last_values[-1] + 1
    except (TypeError, IndexError):
        return count'''

inspect.getsource = fixed_getsource


# Import Process Start Model
try:
    import ProcessStartModel
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

def get_main_config_path():
    """Get main config file path"""
    config_manager = ConfigManager()
    return config_manager.get_config_path('main')


def setup_config_environment():
    """Set up environment variables so other modules can find config files"""
    config_manager = ConfigManager()
    exe_dir = config_manager._get_base_directory()

    # Set environment variables for other modules
    os.environ['CONFIG_BASE_PATH'] = exe_dir
    os.environ['CALIBRATION_CONFIG_PATH'] = config_manager.get_config_path('calibration')
    os.environ['SAM_PARAMETERS_CONFIG_PATH'] = config_manager.get_config_path('sam_parameters')

    logger = get_logger("ConfigSetup")
    logger.info(f"Config environment set up successfully. Base directory: {exe_dir}")


#Set up config environment and get main config path
setup_config_environment()
defaultConfigPath = get_main_config_path()

# Global variables for coordination
processing_queue = queue.Queue()  # For coordinating processing order
processing_lock = threading.Lock()  # For synchronization
current_processing = {}  # Track currently processing samples


config = configparser.ConfigParser()
config.read(get_main_config_path(), encoding='utf-8')

# ==================== Config File Path ====================

FS_DIRECT_PATH = str(config.get('FileSystem', 'DIRECT_PATH', fallback=""))

ERROR_FOLDER=str(config.get('ERROR_FOLDER', 'DIRECT_PATH', fallback=""))

DEST_DIRECT_PATH = str(config.get('Destination', 'DIRECT_PATH', fallback=""))

SAMPLEFOLDER = os.path.abspath(os.path.join(FS_DIRECT_PATH, "Samples"))

# ==================== Frequency of scanning file  ====================
# Get scan interval from config, default to 5 if not found
try:
    SCAN_INTERVAL = config.getint('WORKFLOW', 'SCAN_INTERVAL', fallback=5)
except ValueError:
    logger.warning("Invalid SCAN_INTERVAL in config, using default value: 5")
    SCAN_INTERVAL = 5

# ==================== Sam2 model  ====================
CHECKPOINT_FOLDER = os.path.join(os.getcwd(), "sam2\\checkpoints")
model_name = 'sam2.1_hiera_large.pt'
model_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'


# ==================== Core Analyzing work flow  ====================

def analyze_image(image_path, json_path, checkpoint_folder, FS_DIRECT_PATH):
    """
    Analyze a single image with its corresponding JSON file

    Args:
        image_path: Full path to the image file
        json_path: Full path to the JSON file
        checkpoint_folder: Path to checkpoint folder
        FS_DIRECT_PATH: Path to SAMPLEFOLDER

    Returns:
        newImage: Analyzed image object
    """
    try:
        # Read JSON file and extract ProgramId
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        program_id = json_data.get('ProgramId', 0)
        camera_id = json_data.get('SerialNumber', '')
        # Get image filename (without extension) as sampleID
        sample_id = os.path.basename(image_path)

        # Source folder path: folder created by ProcessStartModel with same name as image

        # Get folder path containing the image (this is picturePath)
        picture_folder = os.path.dirname(image_path)


        # Call ProcessStartModel for analysis
        newImage = ProcessStartModel.ProcessStartModel(
            picturePath=picture_folder,  # The File In  folder containing the image
            sampleID=sample_id,
            programNumber=program_id,
            checkpoint_folder=checkpoint_folder,
            SAMPLEFOLDER=FS_DIRECT_PATH
        )

        # Execute analysis
        newImage.analyse(testing=False)

        logger.info(f"Successfully analyzed: {sample_id}")
        return newImage, sample_id, camera_id

    except Exception as e:
        logger.error(f"Analysis failed for {sample_id}: {e}", exc_info=True)
        raise


def copy_results_to_destination_updated(source_folder_path, sample_id, camera_id):
    """
    Copy analysis results to destination file system with hierarchical folder structure
    Structure: DEST_BASEFOLDER/YYYY/MM - MonthName/DD/CameraID/SampleID
    Uses the original image capture time instead of current time

    Args:
        source_folder_path: Path to the SAMPLEFOLDER/program/sampleID folder
        sample_id: Sample ID for the analysis
        camera_id: Camera ID for organizing files
    Returns:
        bool: True if copy was successful, False otherwise
    """

    perth_tz = pytz.timezone('Australia/Perth')

    try:
        # Try to get image time from BMP file
        bmp_file_path = os.path.join(source_folder_path, f"{sample_id}.bmp")
        file_mtime = os.path.getmtime(bmp_file_path)
        image_time = datetime.fromtimestamp(file_mtime, tz=perth_tz)
    except:
        # If can't get image time, use current time
        image_time = datetime.now(perth_tz)

    # Create folder structure components based on image capture time
    year_folder = image_time.strftime('%Y')
    month_folder = image_time.strftime('%m - %B')  # Format: "10 - October"
    day_folder = image_time.strftime('%d')

    # Build the complete destination path
    dest_path = os.path.join(
        DEST_DIRECT_PATH,
        year_folder,
        month_folder,
        day_folder,
        camera_id,
        sample_id
    )

    # If destination folder already exists, remove it first to avoid conflicts
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)

    # Create destination folder (including all parent directories)
    os.makedirs(dest_path, exist_ok=True)

    # Copy entire sample folder contents
    for item in os.listdir(source_folder_path):
        source_item = os.path.join(source_folder_path, item)
        dest_item = os.path.join(dest_path, item)

        if os.path.isfile(source_item):
            shutil.copy2(source_item, dest_item)
        elif os.path.isdir(source_item):
            shutil.copytree(source_item, dest_item, dirs_exist_ok=True)

    return True

def delete_source_files(image_path, json_path, source_folder_path=None):
    """
    Delete original source files and analysis result folder after successful processing

    Args:
        image_path: Path to image file
        json_path: Path to JSON file
        source_folder_path: Path to the folder containing analysis results (optional)
    """
    try:
        # Delete image file
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.info(f"Deleted image file: {os.path.basename(image_path)}")

        # Delete JSON file
        if os.path.exists(json_path):
            os.remove(json_path)
            logger.info(f"Deleted JSON file: {os.path.basename(json_path)}")

        # Delete source folder with analysis results
        if source_folder_path and os.path.exists(source_folder_path):
            shutil.rmtree(source_folder_path)
            logger.info(f"Deleted source folder: {source_folder_path}")

    except Exception as e:
        logger.error(f"Failed to delete source files: {e}", exc_info=True)
        raise


def transfer_to_error_folder(image_path, json_path, source_folder_path, error_folder):
    """
    Transfer files and analysis folder to error folder

    Args:
        image_path: Path to image file
        json_path: Path to JSON file
        source_folder_path: Path to the folder containing analysis results (can be None)
        error_folder: Error folder path
    """
    try:
        # Create error folder if not exists
        os.makedirs(error_folder, exist_ok=True)

        # Move image file to error folder
        if os.path.exists(image_path):
            shutil.move(image_path, os.path.join(error_folder, os.path.basename(image_path)))
            logger.trace(f"Moved image to error folder: {os.path.basename(image_path)}")
        else:
            logger.trace(f"Image file not found: {image_path}")

        # Move JSON file to error folder
        if os.path.exists(json_path):
            shutil.move(json_path, os.path.join(error_folder, os.path.basename(json_path)))
            logger.trace(f"Moved JSON to error folder: {os.path.basename(json_path)}")
        else:
            logger.trace(f"JSON file not found: {json_path}")

        # Move source folder to error folder (if exists and not None)
        if source_folder_path and os.path.exists(source_folder_path):
            error_folder_destination = os.path.join(error_folder, os.path.basename(source_folder_path))

            # If folder already exists in error folder, remove it first
            if os.path.exists(error_folder_destination):
                shutil.rmtree(error_folder_destination)
                logger.info(f"Removed existing folder in error directory: {os.path.basename(source_folder_path)}")

            shutil.move(source_folder_path, error_folder_destination)
            logger.info(f"Moved analysis folder to error folder: {os.path.basename(source_folder_path)}")
        else:
            if source_folder_path is None:
                logger.info("No analysis folder to move (analysis failed early)")
            else:
                logger.warning(f"Analysis folder not found: {source_folder_path}")

        logger.info(f"Error handling completed for files in: {error_folder}")

    except Exception as e:
        logger.error(f"Failed to move files to error folder: {e}", exc_info=True)
        # Don't raise here to avoid cascading errors


def migrate_all_folders(root_path):
        """Migrate all folders to destination and delete source"""
        for folder_name in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder_name)

            if not os.path.isdir(folder_path):
                continue

            # Find JSON file
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            if not json_files:
                print(f"No JSON in {folder_name}, skipping")
                continue

            # Read camera_id
            with open(os.path.join(folder_path, json_files[0]), 'r') as f:
                camera_id = json.load(f).get('SerialNumber', '')

            # Copy and delete
            try:
                copy_results_to_destination_updated(folder_path, folder_name, camera_id)
                shutil.rmtree(folder_path)
                print(f"Processed: {folder_name}")
            except Exception as e:
                print(f"Error with {folder_name}: {e}")

if __name__ == '__main__':

    # migrate_all_folders(FS_DIRECT_PATH)
    while True:
            files = os.listdir(FS_DIRECT_PATH)

            # Find all .bmp image files
            image_files = [f for f in files if f.endswith('.bmp')]

            if image_files:
                logger.info(f"Found {len(image_files)} image file(s)")

            # Loop through each image file
            for image_file in image_files:
                # Get base name without extension
                base_name = os.path.splitext(image_file)[0]

                # Construct JSON filename
                json_file = base_name + '.json'

                # Check if corresponding JSON file exists
                if json_file not in files:
                    logger.warning(f"JSON file not found for {image_file}, skipping")
                    continue

                # Get full paths
                image_path = os.path.join(FS_DIRECT_PATH, image_file)
                json_path = os.path.join(FS_DIRECT_PATH, json_file)

                logger.info(f"Processing pair: {base_name}")


                picture_folder = os.path.dirname(image_path)
                source_folder_path = os.path.join(picture_folder, base_name)

                try:
                    #Analyze the image
                    result, sample_id, camera_id=analyze_image(
                        image_path,
                        json_path,
                        CHECKPOINT_FOLDER,
                        FS_DIRECT_PATH
                    )


                    #Store image outputs to network share drive

                    copy_results_to_destination_updated(source_folder_path, base_name, '17424009')
                    delete_source_files(image_path, json_path,source_folder_path)


                except Exception as analysis_error:

                    logger.error(f"Analysis error: {analysis_error}", exc_info=True)

                    transfer_to_error_folder(image_path, json_path,source_folder_path, ERROR_FOLDER)

            # Wait before next scan
            time.sleep(SCAN_INTERVAL)

