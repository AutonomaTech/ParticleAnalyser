import sys
import os
import json
import requests
import time
import configparser
import traceback
import threading
import shutil
import subprocess
from logger_config import get_logger
from Config_Manager import ConfigManager

sys.path.append(os.path.join(os.getcwd(), "imageAnalysis"))
sys.path.append(os.path.join(os.getcwd(), 'ImagePreprocessing'))

# if getattr(sys, 'frozen', False):
#     # 打包后：exe文件所在目录
#     exe_dir = os.path.dirname(sys.executable)
# else:
#     # 开发环境：脚本所在目录
#     exe_dir = os.getcwd()
#
# sys.path.append(os.path.join(exe_dir, "ImageAnalysis"))
# sys.path.append(os.path.join(exe_dir, "ImagePreprocessing"))
#
# ProcessStartModel = __import__("ProcessStartModel")


def debug_import():
    """Debug the import process"""
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")

    if hasattr(sys, '_MEIPASS'):
        print(f"Running from PyInstaller, _MEIPASS: {sys._MEIPASS}")
        base_path = sys._MEIPASS
    else:
        print("Running from source")
        base_path = os.path.dirname(os.path.abspath(__file__))

    image_analysis_path = os.path.join(base_path, "ImageAnalysis")
    print(f"Looking for ImageAnalysis at: {image_analysis_path}")
    print(f"ImageAnalysis exists: {os.path.exists(image_analysis_path)}")

    if os.path.exists(image_analysis_path):
        print(f"Contents of ImageAnalysis: {os.listdir(image_analysis_path)}")

    print(f"Current sys.path: {sys.path}")

    # Add to path
    if image_analysis_path not in sys.path:
        sys.path.append(image_analysis_path)
        print(f"Added {image_analysis_path} to sys.path")

    # Try to find ProcessStartModel.py
    process_start_model_path = os.path.join(image_analysis_path, "ProcessStartModel.py")
    print(f"ProcessStartModel.py exists: {os.path.exists(process_start_model_path)}")


# Run debug first
debug_import()

# Now try to import
try:
    import ProcessStartModel

    print("Successfully imported ProcessStartModel")
except ImportError as e:
    print(f"Import error: {e}")
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
    SMB_USERNAME = str(config.get('SMBServer', 'USERNAME', fallback=""))
    SMB_PASSWORD = str(config.get('SMBServer', 'PASSWORD', fallback=""))
    
    # Define the CPU workstation server and folder path
    CPU_SERVER = str(config.get('CPUWorkstation', 'CPU_SERVER', fallback=""))
    CPU_FOLDER = str(config.get('CPUWorkstation', 'CPU_FOLDER', fallback=""))
    CPU_DELETE_AFTER_COPY = config.getboolean('CPUWorkstation', 'DELETE_AFTER_COPY', fallback=True)
    CPU_USERNAME = str(config.get('CPUWorkstation', 'USERNAME', fallback=""))
    CPU_PASSWORD = str(config.get('CPUWorkstation', 'PASSWORD', fallback=""))
    
    # Define the FileSystem configuration (for client deployments)
    FS_ENABLED = config.getboolean('FileSystem', 'ENABLED', fallback=False)
    FS_SERVER = str(config.get('FileSystem', 'SERVER', fallback=""))
    FS_FOLDER = str(config.get('FileSystem', 'FOLDER', fallback=""))
    FS_DIRECT_PATH = str(config.get('FileSystem', 'DIRECT_PATH', fallback=""))
    FS_DELETE_AFTER_COPY = config.getboolean('FileSystem', 'DELETE_AFTER_COPY', fallback=False)
    FS_USERNAME = str(config.get('FileSystem', 'USERNAME', fallback=""))
    FS_PASSWORD = str(config.get('FileSystem', 'PASSWORD', fallback=""))
    
    # Define the Destination configuration (for final results storage)
    DEST_ENABLED = config.getboolean('Destination', 'ENABLED', fallback=False)
    DEST_SERVER = str(config.get('Destination', 'SERVER', fallback=""))
    DEST_FOLDER = str(config.get('Destination', 'FOLDER', fallback=""))
    DEST_DIRECT_PATH = str(config.get('Destination', 'DIRECT_PATH', fallback=""))
    DEST_USERNAME = str(config.get('Destination', 'USERNAME', fallback=""))
    DEST_PASSWORD = str(config.get('Destination', 'PASSWORD', fallback=""))
    
    # Get global credentials as fallback
    GLOBAL_USERNAME = str(config.get('Credentials', 'USERNAME', fallback=""))
    GLOBAL_PASSWORD = str(config.get('Credentials', 'PASSWORD', fallback=""))

except Exception as e:
    logger = get_logger("StartUp")
    logger.error(f"Unexpected error: {e}")
    defaultContainerWidth = 180000
    defaultOutputfolder = "defaultProgram"
    # SMB server defaults
    SMB_SERVER = ""
    SMB_SHARE = "ImageDataShare"
    SMB_USERNAME = ""
    SMB_PASSWORD = ""
    # CPU workstation defaults
    CPU_SERVER = ""
    CPU_FOLDER = ""
    CPU_DELETE_AFTER_COPY = True
    CPU_USERNAME = ""
    CPU_PASSWORD = ""
    # FileSystem defaults
    FS_ENABLED = False
    FS_SERVER = ""
    FS_FOLDER = ""
    FS_DIRECT_PATH = ""
    FS_DELETE_AFTER_COPY = False
    FS_USERNAME = ""
    FS_PASSWORD = ""
    # Destination defaults
    DEST_ENABLED = False
    DEST_SERVER = ""
    DEST_FOLDER = ""
    DEST_DIRECT_PATH = ""
    DEST_USERNAME = ""
    DEST_PASSWORD = ""
    # Global credentials fallback
    GLOBAL_USERNAME = ""
    GLOBAL_PASSWORD = ""

#SAM2
checkpoint_folder = os.path.join(os.getcwd(), "sam2\\checkpoints")
model_name = 'sam2.1_hiera_large.pt'
model_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'

# SMBServer
# SMBServer - 处理本地路径
if SMB_SERVER:
    BASEFOLDER = f"\\\\{SMB_SERVER}\\{SMB_SHARE}"
else:
    BASEFOLDER = SMB_SHARE  # 直接使用本地路径
SAMPLEFOLDER = os.path.abspath(os.path.join(BASEFOLDER, "Samples"))

# CPU Workstation
CPU_BASEFOLDER = f"\\\\{CPU_SERVER}\\{CPU_FOLDER}" if CPU_SERVER and CPU_FOLDER else ""

# FileSystem (for client deployments)
if FS_ENABLED:
    FS_BASEFOLDER = FS_DIRECT_PATH if FS_DIRECT_PATH else (f"\\\\{FS_SERVER}\\{FS_FOLDER}" if FS_SERVER and FS_FOLDER else "")
else:
    FS_BASEFOLDER = ""

# Destination (for final results)
if DEST_ENABLED:
    DEST_BASEFOLDER = DEST_DIRECT_PATH if DEST_DIRECT_PATH else (f"\\\\{DEST_SERVER}\\{DEST_FOLDER}" if DEST_SERVER and DEST_FOLDER else "")
else:
    DEST_BASEFOLDER = ""

# Logger setup
logger = get_logger("StartUp")

def analyze_folder(folder_path):
    """ Continuously analyze files in the folder for BMP and corresponding JSON files. """
    while True:
        logger.info("Monitoring GPU server folder...")
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

                            newImage.analyse(testing=False)
                            updateStatusJson()
                            
                            # Handle destination copy and cleanup if enabled
                            if DEST_ENABLED:
                                try:
                                    # Get the result folder path from ProcessStartModel
                                    result_folder_path = newImage.picturePath  # This is the SAMPLEFOLDER/program/sampleID path
                                    sample_id = newImage.sampleID
                                    program_number = newImage.programNumber if newImage.programNumber not in [None, 0] else "defaultProgram"
                                    
                                    logger.info(f"Starting destination copy for sample: {sample_id}")
                                    
                                    # Copy results to destination
                                    copy_success = copy_results_to_destination(result_folder_path, sample_id, str(program_number))
                                    
                                    if copy_success:
                                        logger.info(f"Destination copy successful for {sample_id}")
                                        
                                        # Read source file info for cleanup
                                        source_info_file = bmp_file + ".source_info"
                                        if os.path.exists(source_info_file):
                                            try:
                                                import json
                                                with open(source_info_file, 'r') as f:
                                                    source_info = json.load(f)
                                                
                                                source_bmp = source_info.get("source_bmp")
                                                source_json = source_info.get("source_json")
                                                source_type = source_info.get("source_type", "Unknown")
                                                
                                                # Only cleanup if we didn't delete source files during copy
                                                if source_bmp and source_json and os.path.exists(source_bmp):
                                                    logger.info(f"Cleaning up source files from {source_type}")
                                                    cleanup_success = cleanup_after_destination_copy(
                                                        source_bmp, source_json, result_folder_path
                                                    )
                                                    
                                                    if cleanup_success:
                                                        logger.info(f"Successfully cleaned up source files for {sample_id}")
                                                    else:
                                                        logger.warning(f"Partial cleanup failure for {sample_id}")
                                                else:
                                                    logger.info(f"Source files already deleted or not found, only cleaning SMB result folder")
                                                    # Still clean up the SMB result folder
                                                    try:
                                                        if os.path.exists(result_folder_path):
                                                            shutil.rmtree(result_folder_path)
                                                            logger.info(f"Cleaned up SMB result folder: {result_folder_path}")
                                                    except Exception as e:
                                                        logger.error(f"Error cleaning up SMB result folder: {e}")
                                                
                                                # Clean up the source info file
                                                os.remove(source_info_file)
                                                logger.debug(f"Removed source info file: {source_info_file}")
                                                
                                            except Exception as e:
                                                logger.error(f"Error reading source info file: {e}")
                                        else:
                                            logger.warning(f"Source info file not found: {source_info_file}")
                                            # Still clean up the SMB result folder
                                            try:
                                                if os.path.exists(result_folder_path):
                                                    shutil.rmtree(result_folder_path)
                                                    logger.info(f"Cleaned up SMB result folder: {result_folder_path}")
                                            except Exception as e:
                                                logger.error(f"Error cleaning up SMB result folder: {e}")
                                    else:
                                        logger.error(f"Destination copy failed for {sample_id}, keeping all files")
                                        
                                except Exception as e:
                                    logger.error(f"Error in destination handling: {e}")
                            else:
                                logger.debug("Destination not enabled, keeping all files")
                                
                        except Exception as e:
                            logger.error(f"Error processing {bmp_file} and {json_file}: {str(e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                    else:
                        logger.error(f"Missing JSON file for {bmp_file}")
                        continue

        time.sleep(1)

def monitor_external_folder(folder_path, delete_after_copy=True, source_type="CPU", username=None, password=None):
    """ 
    Continuously monitor an external folder and its subfolders for BMP and corresponding JSON files.
    
    Args:
        folder_path: Path to the folder to monitor
        delete_after_copy: Whether to delete files after copying them
        source_type: Type of source ("CPU" or "FileSystem") for logging
        username: Username for network share authentication
        password: Password for network share authentication
    """
    if not folder_path:
        logger.error(f"{source_type} configuration is missing. Cannot monitor {source_type}.")
        logger.error(f"ERROR: {source_type} configuration is missing or empty. Cannot monitor.")
        return
    
    logger.info(f"Starting {source_type} monitoring at {folder_path}")
    logger.info(f"Starting {source_type} monitoring at {folder_path}")
    
    # Try to establish the connection first
    if username and password:
        connect_network_share(folder_path, username, password)
    elif GLOBAL_USERNAME and GLOBAL_PASSWORD:
        # Fall back to global credentials if specific ones aren't provided
        connect_network_share(folder_path, GLOBAL_USERNAME, GLOBAL_PASSWORD)
    
    # Cache to keep track of processed files to avoid re-processing
    processed_files = set()
    
    while True:
        try:
            logger.info(f"Monitoring {source_type} folder and subfolders: {folder_path}...")
            
            # Function to find all BMP files in the folder and subfolders
            def find_bmp_files(base_folder):
                bmp_files = []
                try:
                    # Process files in the main folder
                    for filename in os.listdir(base_folder):
                        full_path = os.path.join(base_folder, filename)
                        
                        if os.path.isfile(full_path) and filename.endswith('.bmp'):
                            bmp_files.append((full_path, filename))
                        
                        # If this is a subfolder, recursively scan it
                        elif os.path.isdir(full_path):
                            subfolder_files = find_bmp_files(full_path)
                            bmp_files.extend(subfolder_files)
                            
                except Exception as e:
                    logger.error(f"Error accessing directory {base_folder}: {str(e)}")
                
                return bmp_files
            
            # Find all BMP files
            try:
                all_bmp_files = find_bmp_files(folder_path)
                logger.info(f"Found {len(all_bmp_files)} BMP files in total")
                
                # Process each BMP file if it has a corresponding JSON
                for bmp_file_path, bmp_filename in all_bmp_files:
                    # Skip if we've already processed this file
                    if bmp_file_path in processed_files:
                        continue
                    
                    folder_path = os.path.dirname(bmp_file_path)
                    json_file = bmp_file_path.replace('.bmp', '.json')
                    
                    if os.path.exists(json_file):
                        try:
                            logger.info(f"Found matching pair: {bmp_file_path} and {json_file}")
                            
                            # Define destination paths
                            dest_bmp = os.path.join(BASEFOLDER, bmp_filename)
                            dest_json = os.path.join(BASEFOLDER, bmp_filename.replace('.bmp', '.json'))
                            
                            # Simple file copy operations
                            logger.info(f"Copying {bmp_file_path} to {dest_bmp}")
                            shutil.copy2(bmp_file_path, dest_bmp)
                            
                            logger.info(f"Copying {json_file} to {dest_json}")
                            shutil.copy2(json_file, dest_json)
                            
                            logger.info(f"Copied {bmp_filename} and its JSON from {source_type} to GPU server")
                            logger.info(f"SUCCESS: Copied {bmp_filename} and its JSON to GPU server")
                            
                            # Save source file paths for later cleanup (if destination is enabled)
                            if DEST_ENABLED:
                                source_info_file = dest_bmp + ".source_info"
                                source_info = {
                                    "source_bmp": bmp_file_path,
                                    "source_json": json_file,
                                    "source_type": source_type,
                                    "copy_time": time.time()
                                }
                                import json
                                with open(source_info_file, 'w') as f:
                                    json.dump(source_info, f)
                                logger.debug(f"Saved source info for cleanup: {source_info_file}")
                            
                            # Delete original files after successful copy if configured to do so
                            if delete_after_copy:
                                os.remove(bmp_file_path)
                                os.remove(json_file)
                                logger.info(f"Removed original files from {source_type}")
                                logger.info(f"SUCCESS: Removed original files after successful copy")
                            else:
                                logger.info(f"Kept original files on {source_type} (deletion disabled)")
                                logger.info(f"INFO: Kept original files (deletion disabled)")
                            
                            # Add to processed files cache
                            processed_files.add(bmp_file_path)
                            
                            # Clean up the cache occasionally to prevent memory growth
                            if len(processed_files) > 1000:
                                processed_files.clear()  # Reset after a large number to prevent memory issues
                                
                        except Exception as e:
                            logger.error(f"Error copying files: {str(e)}")
                            logger.error(f"ERROR copying files: {str(e)}")
                    else:
                        logger.info(f"No matching JSON file found for {bmp_file_path}")
            except Exception as e:
                logger.error(f"Error scanning for files: {str(e)}")
                # Try to reconnect if access fails, using the appropriate credentials
                if username and password:
                    logger.info(f"Attempting to reconnect to {source_type} network share...")
                    connect_network_share(folder_path, username, password)
                elif GLOBAL_USERNAME and GLOBAL_PASSWORD:
                    logger.info(f"Attempting to reconnect to {source_type} network share using global credentials...")
                    connect_network_share(folder_path, GLOBAL_USERNAME, GLOBAL_PASSWORD)
                time.sleep(10)
                continue
                
        except Exception as e:
            logger.error(f"Error in {source_type} monitoring loop: {str(e)}")
            logger.error(f"ERROR in {source_type} monitoring loop: {str(e)}")
        
        time.sleep(5)  # Check every 5 seconds

def monitor_cpu_workstation():
    """ Wrapper for monitoring CPU workstation """
    monitor_external_folder(CPU_BASEFOLDER, CPU_DELETE_AFTER_COPY, "CPU Workstation", CPU_USERNAME, CPU_PASSWORD)

def monitor_filesystem():
    """ Wrapper for monitoring the client's file system """
    monitor_external_folder(FS_BASEFOLDER, FS_DELETE_AFTER_COPY, "FileSystem", FS_USERNAME, FS_PASSWORD)

def updateStatusJson():
    machineStatusJson=os.path.abspath(os.path.join(BASEFOLDER, "machineStatus.json"))

    try:
        with open(machineStatusJson, 'r') as f:
            data = json.load(f)

        # Check if 'ProcessCount' key exists
        if 'ProcessCount' in data:
            # Decrease the ProcessCount by 1
            data['ProcessCount'] = max(0, data['ProcessCount'] - 1)

            logger.info(f"Updated ProcessCount in {machineStatusJson}: {data['ProcessCount']}")

        else:
            data['ProcessCount'] = 0
            
        with open(machineStatusJson, 'w') as f:
            json.dump(data, f, indent=4)

    except Exception as e:
        logger.error(f"Error updating status file {machineStatusJson}: {e}")

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

def check_remote_folder_available(remote_folder, timeout=10):
    """Check if the remote folder is accessible and openable with a timeout."""
    logger.info(f"Checking if remote folder {remote_folder} is accessible...")
    
    try:
        # Use a timeout mechanism to prevent hanging
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def check_folder():
            try:
                # First, check if the folder exists
                folder_exists = os.path.exists(remote_folder)
                if not folder_exists:
                    result_queue.put((False, f"Remote folder {remote_folder} does not exist."))
                    return
                
                # Check if it's a directory
                if not os.path.isdir(remote_folder):
                    result_queue.put((False, f"The path {remote_folder} is not a directory."))
                    return
                
                # Try to list the contents of the folder
                try:
                    files = os.listdir(remote_folder)
                    result_queue.put((True, f"Successfully accessed {remote_folder} and found {len(files)} files/folders."))
                except PermissionError:
                    result_queue.put((False, f"PermissionError: Unable to access the folder {remote_folder}. Check permissions."))
                except OSError as e:
                    result_queue.put((False, f"OSError: Unable to access the folder {remote_folder}. Error: {e}"))
            except Exception as e:
                result_queue.put((False, f"Error accessing remote folder {remote_folder}: {e}"))
        
        # Start the check in a separate thread
        check_thread = threading.Thread(target=check_folder)
        check_thread.daemon = True
        check_thread.start()
        
        # Wait for the result with timeout
        try:
            success, message = result_queue.get(timeout=timeout)
            logger.info(message)
            return success
        except queue.Empty:
            logger.error(f"TIMEOUT: Check for remote folder {remote_folder} took too long (>{timeout}s). Network path may be unreachable.")
            return False
            
    except Exception as e:
        logger.error(f"Error in check_remote_folder_available: {e}")
        return False

def create_remote_folder(remote_folder, max_retries=5):
    """Checks if remote folder is available, retries up to max_retries times."""
    logger.info(f"Checking if remote folder {remote_folder} is available...")
    retries = 0
    success = False
    
    while retries < max_retries:
        success = check_remote_folder_available(remote_folder, timeout=10)
        if success:
            logger.info(f"Remote folder {remote_folder} is now available.")
            return True
        
        retries += 1
        if retries < max_retries:
            logger.info(f"Remote folder {remote_folder} not available. Retrying... Attempt {retries}/{max_retries}")
            time.sleep(5)  # Retry after 5 seconds
    
    if not success:
        logger.warning(f"WARNING: Could not connect to {remote_folder} after {max_retries} attempts.")
        logger.warning(f"Will continue with the program, but functionality may be limited.")
    
    return success

def connect_network_share(network_path, username=None, password=None):
    """Connect to a network share using explicit credentials if provided."""
    try:
        if not username or not password:
            logger.info(f"Attempting to connect to {network_path} with current user credentials")
            return True
            
        logger.info(f"Attempting to connect to {network_path} with provided credentials")
        
        # Use the NET USE command to connect with credentials
        cmd = f'NET USE "{network_path}" /USER:{username} "{password}"'
        
        # Execute the command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Check if connection was successful
        if result.returncode == 0:
            logger.info(f"Successfully connected to {network_path}")
            return True
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            logger.error(f"Failed to connect to {network_path}: {error_msg}")
            # Try to connect without credentials as fallback
            logger.info("Trying to connect without explicit credentials...")
            return True
            
    except Exception as e:
        logger.error(f"Error connecting to network share: {e}")
        return False

def copy_results_to_destination(source_folder_path, sample_id, program_number):
    """
    Copy analysis results to destination file system
    
    Args:
        source_folder_path: Path to the SAMPLEFOLDER/program/sampleID folder
        sample_id: Sample ID for the analysis
        program_number: Program number for organization
        
    Returns:
        bool: True if copy was successful, False otherwise
    """
    if not DEST_ENABLED or not DEST_BASEFOLDER:
        logger.info("Destination not enabled or configured, skipping copy")
        return True  # Consider this as "success" since it's not required
    
    try:
        # Create destination path structure matching source
        dest_program_folder = os.path.join(DEST_BASEFOLDER, program_number if program_number not in [None, 0] else "defaultProgram")
        dest_sample_folder = os.path.join(dest_program_folder, sample_id)
        
        # Ensure destination folders exist
        os.makedirs(dest_sample_folder, exist_ok=True)
        logger.info(f"Created destination folder: {dest_sample_folder}")
        
        # Copy the entire sample folder
        logger.info(f"Copying results from {source_folder_path} to {dest_sample_folder}")
        
        # Copy all files from source to destination
        for item in os.listdir(source_folder_path):
            source_item = os.path.join(source_folder_path, item)
            dest_item = os.path.join(dest_sample_folder, item)
            
            if os.path.isfile(source_item):
                shutil.copy2(source_item, dest_item)
                logger.debug(f"Copied file: {item}")
            elif os.path.isdir(source_item):
                shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                logger.debug(f"Copied directory: {item}")
        
        logger.info(f"Successfully copied results to destination: {dest_sample_folder}")
        
        # Verify the copy was successful
        return verify_destination_copy(source_folder_path, dest_sample_folder)
        
    except Exception as e:
        logger.error(f"Error copying results to destination: {e}")
        return False

def verify_destination_copy(source_folder, dest_folder):
    """
    Verify that all files were successfully copied to destination
    
    Args:
        source_folder: Source folder path
        dest_folder: Destination folder path
        
    Returns:
        bool: True if verification successful, False otherwise
    """
    try:
        logger.info(f"Verifying copy from {source_folder} to {dest_folder}")
        
        # Get list of all files in source
        source_files = []
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), source_folder)
                source_files.append(rel_path)
        
        # Check if all files exist in destination and have same size
        for rel_path in source_files:
            source_file = os.path.join(source_folder, rel_path)
            dest_file = os.path.join(dest_folder, rel_path)
            
            # Check if destination file exists
            if not os.path.exists(dest_file):
                logger.error(f"Destination file missing: {rel_path}")
                return False
            
            # Check if file sizes match
            source_size = os.path.getsize(source_file)
            dest_size = os.path.getsize(dest_file)
            
            if source_size != dest_size:
                logger.error(f"File size mismatch for {rel_path}: source={source_size}, dest={dest_size}")
                return False
        
        logger.info(f"Verification successful: {len(source_files)} files verified")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying destination copy: {e}")
        return False

def cleanup_after_destination_copy(source_bmp_path, source_json_path, smb_result_folder):
    """
    Clean up source files and SMB temp files after successful destination copy
    
    Args:
        source_bmp_path: Original BMP file path in source FS
        source_json_path: Original JSON file path in source FS  
        smb_result_folder: Result folder path in SAMPLEFOLDER to clean up
        
    Returns:
        bool: True if cleanup successful, False otherwise
    """
    cleanup_success = True
    
    try:
        # Delete source files in FS
        if os.path.exists(source_bmp_path):
            os.remove(source_bmp_path)
            logger.info(f"Deleted source BMP file: {source_bmp_path}")
        
        if os.path.exists(source_json_path):
            os.remove(source_json_path)
            logger.info(f"Deleted source JSON file: {source_json_path}")
            
    except Exception as e:
        logger.error(f"Error deleting source files: {e}")
        cleanup_success = False
    
    try:
        # Clean up SMB result folder
        if os.path.exists(smb_result_folder):
            shutil.rmtree(smb_result_folder)
            logger.info(f"Cleaned up SMB result folder: {smb_result_folder}")
            
    except Exception as e:
        logger.error(f"Error cleaning up SMB result folder: {e}")
        # SMB cleanup failure is not critical, don't set cleanup_success to False
        logger.warning("SMB cleanup failed but continuing...")
    
    return cleanup_success

if __name__ == '__main__':    
    # First try to connect to network shares using appropriate credentials
    logger.info("Attempting to connect to network shares...")
    
    # Connect to GPU server using SMB credentials or global fallback
    if SMB_USERNAME and SMB_PASSWORD:
        connect_network_share(BASEFOLDER, SMB_USERNAME, SMB_PASSWORD)
    elif GLOBAL_USERNAME and GLOBAL_PASSWORD:
        connect_network_share(BASEFOLDER, GLOBAL_USERNAME, GLOBAL_PASSWORD)
    
    # Connect to CPU workstation if configured
    if CPU_BASEFOLDER:
        if CPU_USERNAME and CPU_PASSWORD:
            connect_network_share(CPU_BASEFOLDER, CPU_USERNAME, CPU_PASSWORD)
        elif GLOBAL_USERNAME and GLOBAL_PASSWORD:
            connect_network_share(CPU_BASEFOLDER, GLOBAL_USERNAME, GLOBAL_PASSWORD)
    
    # Connect to FileSystem if configured and enabled
    if FS_ENABLED and FS_BASEFOLDER:
        if FS_USERNAME and FS_PASSWORD:
            connect_network_share(FS_BASEFOLDER, FS_USERNAME, FS_PASSWORD)
        elif GLOBAL_USERNAME and GLOBAL_PASSWORD:
            connect_network_share(FS_BASEFOLDER, GLOBAL_USERNAME, GLOBAL_PASSWORD)
    
    # Connect to Destination if configured and enabled
    if DEST_ENABLED and DEST_BASEFOLDER:
        if DEST_USERNAME and DEST_PASSWORD:
            connect_network_share(DEST_BASEFOLDER, DEST_USERNAME, DEST_PASSWORD)
        elif GLOBAL_USERNAME and GLOBAL_PASSWORD:
            connect_network_share(DEST_BASEFOLDER, GLOBAL_USERNAME, GLOBAL_PASSWORD)
    
    # Try to connect to the GPU server, but continue even if it fails
    smb_available = create_remote_folder(BASEFOLDER)
    
    if smb_available:
        if not os.path.exists(SAMPLEFOLDER):
            try:
                os.makedirs(SAMPLEFOLDER)
                logger.info(f"Local folder {SAMPLEFOLDER} created.")
            except Exception as e:
                logger.error(f"Error creating SAMPLEFOLDER: {e}")
                logger.error(f"Error creating SAMPLEFOLDER: {e}")
        else:
            logger.info(f"Local folder {SAMPLEFOLDER} already exists.")
    else:
        logger.warning(f"WARNING: GPU server folder {BASEFOLDER} not available. Analysis will be limited.")

    # Try to download the model, but continue if it fails
    try:
        download_model()
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        logger.error(f"Error downloading model: {e}")
    
    # Check destination availability if enabled
    dest_available = False
    if DEST_ENABLED and DEST_BASEFOLDER:
        logger.info(f"Destination enabled: {DEST_BASEFOLDER}")
        dest_available = create_remote_folder(DEST_BASEFOLDER)
        if dest_available:
            logger.info("Destination folder is available and accessible")
        else:
            logger.warning("Destination folder is not accessible - destination copy will fail")
    elif DEST_ENABLED:
        logger.warning("Destination enabled but no valid path configured")
    else:
        logger.info("Destination not enabled - analysis results will be kept in SAMPLEFOLDER only")
    
    # Start monitoring threads based on configuration
    monitor_threads = []
    
    # Check if FileSystem monitoring is enabled (takes precedence)
    if FS_ENABLED and FS_BASEFOLDER:
        logger.info(f"FileSystem monitoring enabled: {FS_BASEFOLDER}")
        logger.info(f"File deletion after copy: {'ENABLED' if FS_DELETE_AFTER_COPY else 'DISABLED'}")
        
        # Start FileSystem monitoring thread
        fs_thread = threading.Thread(target=monitor_filesystem, daemon=True)
        fs_thread.start()
        monitor_threads.append(fs_thread)
        logger.info("Started FileSystem monitoring thread")
    # Otherwise use CPU workstation if configured
    elif CPU_BASEFOLDER:
        logger.info(f"CPU workstation configuration detected: {CPU_BASEFOLDER}")
        logger.info(f"File deletion after copy: {'ENABLED' if CPU_DELETE_AFTER_COPY else 'DISABLED'}")
        
        # Start CPU workstation monitoring thread
        cpu_thread = threading.Thread(target=monitor_cpu_workstation, daemon=True)
        cpu_thread.start()
        monitor_threads.append(cpu_thread)
        logger.info("Started CPU workstation monitoring thread")
    else:
        logger.info("No external monitoring configuration found. No monitoring threads started.")
    
    # Only start analyzing the GPU server folder if it's available
    if smb_available:
        # Start the main analysis function (this will run in the main thread)
        logger.info(f"Starting analysis on GPU server folder: {BASEFOLDER}")
        analyze_folder(BASEFOLDER)
    else:
        # If GPU server is not available but we have monitoring threads running,
        # keep the main thread alive to allow monitoring to continue
        if monitor_threads:
            logger.info("GPU server folder not available. Only external monitoring will be active.")
            while True:
                time.sleep(60)
                logger.info("Main thread still alive, monitoring threads active...")
        else:
            logger.info("No monitoring configuration and GPU server not available. Nothing to do.")
            logger.info("Exiting program.")
            
