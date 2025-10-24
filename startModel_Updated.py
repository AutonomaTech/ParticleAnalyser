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


# Set up config environment and get main config path
setup_config_environment()
defaultConfigPath = get_main_config_path()

# Global variables for coordination
processing_queue = queue.Queue()  # For coordinating processing order
processing_lock = threading.Lock()  # For synchronization
current_processing = {}  # Track currently processing samples

config = configparser.ConfigParser()
config.read(get_main_config_path(), encoding='utf-8')

# ==================== Config File Path ====================

FS_DIRECT_PATH = str(config.get('FileIn', 'DIRECT_PATH', fallback=""))

FS_Out_DIRECT_PATH = str(config.get('FileOut', 'DIRECT_PATH', fallback=""))

ERROR_FOLDER = str(config.get('ERROR_FOLDER', 'DIRECT_PATH', fallback=""))

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
            picturePath=picture_folder,  # The Temp  folder containing the image
            jsonPath=json_path,
            sampleID=sample_id,
            programNumber=program_id,
            checkpoint_folder=checkpoint_folder,
            SAMPLEFOLDER=FS_Out_DIRECT_PATH
        )

        # Execute analysis
        newImage.analyse(testing=False)

        logger.info(f"Successfully analyzed: {sample_id}")
        return newImage, sample_id, camera_id

    except Exception as e:
        logger.error(f"Analysis failed for {sample_id}: {e}", exc_info=True)
        # return None, sample_id, "test"
        raise e


def handle_successful_analysis_and_cleanup(temp_folder_path, base_name, camera_id,
                                           image_path_filein, json_path_filein,
                                           image_file, json_file):
    """
    Handle successful analysis: copy to destination and cleanup all files

    Args:
        temp_folder_path: Path to temporary folder in FileOut
        base_name: Base name of the sample
        camera_id: Camera ID for organizing files
        image_path_filein: Path to image file in FileIn
        json_path_filein: Path to JSON file in FileIn
        image_file: Image filename
        json_file: JSON filename

    Returns:
        bool: True if copy succeeded, False if copy failed (but cleanup still done)
    """
    # Try to copy results to destination
    try:
        copy_results_to_destination_updated(temp_folder_path, base_name, camera_id)
        logger.info(f"Successfully copied to destination: {base_name}")

        # If copy successful: delete temp folder AND delete FileIn files
        if os.path.exists(temp_folder_path):
            shutil.rmtree(temp_folder_path)
            logger.info(f"Deleted temp folder: {temp_folder_path}")

        # Delete original files in FileIn
        if os.path.exists(image_path_filein):
            os.remove(image_path_filein)
            logger.info(f"Deleted FileIn image: {image_file}")

        if os.path.exists(json_path_filein):
            os.remove(json_path_filein)
            logger.info(f"Deleted FileIn JSON: {json_file}")

        return True

    except Exception as copy_error:
        # Analysis succeeded but copy to destination failed
        logger.error(f"Failed to copy {base_name} to destination: {copy_error}", exc_info=True)

        # Delete temp folder only (keep FileIn files)
        if os.path.exists(temp_folder_path):
            shutil.rmtree(temp_folder_path)
            logger.info(f"Deleted temp folder after copy failure: {temp_folder_path}")

        return False
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
    # raise Exception(f"Simulated copy to destination failure for sample: {sample_id}")
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


if __name__ == '__main__':

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

            # Get full paths in FileIn
            image_path_filein = os.path.join(FS_DIRECT_PATH, image_file)
            json_path_filein = os.path.join(FS_DIRECT_PATH, json_file)

            logger.info(f"Processing pair: {base_name}")

            # Create temporary folder in FileOut
            temp_folder_path = os.path.join(FS_Out_DIRECT_PATH, base_name)

            try:
                # Create temp folder in FileOut
                os.makedirs(temp_folder_path, exist_ok=True)

                # Copy image and JSON to temp folder
                image_path_temp = os.path.join(temp_folder_path, image_file)
                json_path_temp = os.path.join(temp_folder_path, json_file)

                shutil.copy2(image_path_filein, image_path_temp)
                shutil.copy2(json_path_filein, json_path_temp)
                logger.info(f"Created temp folder and copied files to: {temp_folder_path}")

                # Analyze the image (using temp folder paths)
                result, sample_id, camera_id = analyze_image(
                    image_path_temp,
                    json_path_temp,
                    CHECKPOINT_FOLDER,
                    FS_Out_DIRECT_PATH
                )

                logger.info(f"Analysis completed successfully for: {base_name}")

                # Handle successful analysis: copy to destination and cleanup
                handle_successful_analysis_and_cleanup(
                    temp_folder_path, base_name, camera_id,
                    image_path_filein, json_path_filein,
                    image_file, json_file
                )

            except Exception as analysis_error:
                # Check if this is a CUDA out of memory error
                error_message = str(analysis_error).lower()
                is_cuda_oom = 'cuda out of memory' in error_message or 'out of memory' in error_message

                if is_cuda_oom:
                    # Treat CUDA OOM as successful analysis - copy to destination and clean up all files
                    logger.warning(f"CUDA out of memory for {base_name}, treating as successful and cleaning up files")

                    # Try to get camera_id from JSON file
                    try:
                        with open(json_path_temp, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        camera_id = json_data.get('SerialNumber', 'unknown')
                    except:
                        camera_id = 'unknown'

                    # Handle as successful analysis: copy to destination and cleanup
                    handle_successful_analysis_and_cleanup(
                        temp_folder_path, base_name, camera_id,
                        image_path_filein, json_path_filein,
                        image_file, json_file
                    )
                else:
                    # Regular analysis failure - move to error folder
                    logger.error(f"Analysis failed for {base_name}: {analysis_error}", exc_info=True)

                    # Move temp folder to error folder
                    try:
                        os.makedirs(ERROR_FOLDER, exist_ok=True)
                        error_destination = os.path.join(ERROR_FOLDER, base_name)

                        # Remove existing error folder if exists
                        if os.path.exists(error_destination):
                            shutil.rmtree(error_destination)

                        # Move temp folder to error folder
                        if os.path.exists(temp_folder_path):
                            shutil.move(temp_folder_path, error_destination)
                            logger.info(f"Moved temp folder to error folder: {base_name}")

                        # Delete FileIn files after moving to error
                        if os.path.exists(image_path_filein):
                            os.remove(image_path_filein)
                            logger.info(f"Deleted FileIn image after error: {image_file}")

                        if os.path.exists(json_path_filein):
                            os.remove(json_path_filein)
                            logger.info(f"Deleted FileIn JSON after error: {json_file}")

                    except Exception as error_handling_exception:
                        logger.error(f"Failed to handle error for {base_name}: {error_handling_exception}",
                                     exc_info=True)

        # Wait before next scan
        time.sleep(SCAN_INTERVAL)
