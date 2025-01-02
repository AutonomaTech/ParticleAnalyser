import os
import subprocess
import argparse
import configparser
import traceback
from logger_config import get_logger
logger = get_logger("AutoProcess")


def check_and_run_command(samples_root_folder):
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        container_width = int(config['analysis']['containerWidth'])
        # Traverse through each main sample folder
        for folder in os.listdir(samples_root_folder):

            sample_folder = os.path.join(samples_root_folder, folder)
            logger.info(f"Processing all samples within the folder: {sample_folder}")
            if os.path.isdir(sample_folder):
                # Traverse each image sub-folder inside the sample folder
                for subfolder in os.listdir(sample_folder):
                    image_folder_path = os.path.join(sample_folder, subfolder)
                    logger.info(f"Processing : {sample_folder}")

                    if os.path.isdir(image_folder_path):
                        # Check for files ending with '_distribution.txt'
                        # distribution_files = [f for f in os.listdir(image_folder_path) if f.endswith('_distribution.txt')]

                        # if not distribution_files:
                            # If no distribution file is found, run the command
                            command = f"python startup.py {image_folder_path} --containerWidth={container_width}"
                            print(f"Running command: {command}")
                            subprocess.run(command, shell=True)
    except Exception as e:
        logger.error(f"Fatal error in  batch auto process: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")



def main(samples_root_folder):
    """
    Main function to process all samples within the given root folder.
    """
    logger.info(f"Processing all samples within the folder: {samples_root_folder}")
    check_and_run_command(samples_root_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process samples and run commands based on specific conditions.")
    parser.add_argument('samples_root_folder', type=str, help='Root folder path containing all sample subfolders.')

    args = parser.parse_args()

    main(args.samples_root_folder)
