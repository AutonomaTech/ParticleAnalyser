# tester_flow.py

import json
from datetime import datetime
import os
import sys
import argparse
import requests
import logging

# Importing the ImageAnalysisModel from the analyser_module package
import ImageAnalysisModel as pa


def download_model(checkpoint_folder, file_url, file_name):
    """
    Checks if the SAM model exists; if not, downloads it.

    Args:
        checkpoint_folder (str): Path to the folder where the model will be stored.
        file_url (str): URL to download the model from.
        file_name (str): Name of the model file.
    """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)
        logging.info(f"Checkpoints folder created: {checkpoint_folder}")

    file_path = os.path.join(checkpoint_folder, file_name)
    if not os.path.exists(file_path):
        try:
            logging.info(f"Downloading model file: {file_name}...")
            response = requests.get(file_url, stream=True)
            response.raise_for_status()  # Check if the request was successful
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logging.info("SAM model downloaded successfully.")
        except Exception as e:
            logging.error(f"Error occurred during SAM model download: {e}")
            sys.exit(1)
    else:
        logging.info("Checkpoint already exists, skipping download.")


def load_config(config_path):
    """
    Loads the parameters configuration JSON file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        dict: Configuration parameters.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main(image_folder_path, config_path, checkpoint_folder, model_url, model_name, containerWidth, scalingNumber):
    """
    Main function to execute the automated testing flow.

    Args:
        image_folder_path (str): Path to the folder containing images for analysis.
        config_path (str): Path to the parameters configuration JSON file.
        checkpoint_folder (str): Folder to store/download the SAM model checkpoints.
        model_url (str): URL to download the SAM model.
        model_name (str): Name of the SAM model file.
        containerWidth (float): Width of the container used for scaling (in um).
        scalingNumber (float): Scaling number used for scaling factor calculation.
    """
    # Load configuration file
    config = load_config(config_path)
    parameter_sets = config.get('parameter_sets', [])
    bins = config.get('bins', [])

    if not parameter_sets:
        logging.error("No parameter sets found in the configuration file.")
        sys.exit(1)

    if not bins:
        logging.error("No bins found in the configuration file.")
        sys.exit(1)

    # Download the SAM model if necessary
    download_model(checkpoint_folder, model_url, model_name)

    # Iterate through each parameter set
    for idx, params in enumerate(parameter_sets):
        try:
            # Create a unique folder name for the current parameter set
            parameter_folder_name = f"parameter_{idx + 1}"
            logging.info(f"\n=== Processing Parameter Set {idx + 1}/{len(parameter_sets)} ===")

            # Instantiate ImageAnalysisModel with the provided image folder path
            analyser =pa. ImageAnalysisModel(
                image_folder_path=image_folder_path,
                scalingNumber=scalingNumber,
                containerWidth=containerWidth,
            )

            # Perform preprocessing operations
            analyser.evenLightingWithValidation(parameter_folder_name)
            analyser.overlayImageWithValidation()

            # Record the start time
            start_time = datetime.now()
            logging.info(f"Analyzing particles {idx + 1}/{len(parameter_sets)}, Start Time: {start_time}")

            # Analyze particles using the current parameter set
            analyser.analyseValidationParticles(
                checkpoint_folder=checkpoint_folder,
                parameter_folder_name=parameter_folder_name,
                testing_parameters=params
            )

            # Save analysis results
            analyser.saveSegments()
            analyser.saveResultsForValidation(bins, parameter_folder_name)
            analyser.formatResults()

            # Record the end time
            end_time = datetime.now()
            logging.info(f"Completed particle analysis: {idx + 1}/{len(parameter_sets)}, End Time: {end_time}")
            logging.info(f"Total Time: {end_time - start_time}")

        except Exception as e:
            logging.error(f"Error processing parameter set {idx + 1}: {e}")
            continue


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("tester_flow.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Automated Testing Flow for Image Analysis")
    parser.add_argument(
        '--image_folder_path',
        type=str,
        required=True,
        help='Path to the folder containing images for analysis.'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='parameters_config.json',
        help='Path to the parameters configuration JSON file.'
    )
    parser.add_argument(
        '--checkpoint_folder',
        type=str,
        default='checkpoints',
        help='Folder to store/download the SAM model checkpoints.'
    )
    parser.add_argument(
        '--model_url',
        type=str,
        default='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
        help='URL to download the SAM model.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='sam2.1_hiera_large.pt',
        help='Name of the SAM model file.'
    )
    parser.add_argument(
        '--containerWidth',
        type=float,
        default=1.0,
        help='Width of the container used for scaling (in um).'
    )
    parser.add_argument(
        '--scalingNumber',
        type=float,
        default=1.0,
        help='Scaling number used for scaling factor calculation.'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Execute the main function with the provided arguments
    main(
        image_folder_path=args.image_folder_path,
        config_path=args.config_path,
        checkpoint_folder=args.checkpoint_folder,
        model_url=args.model_url,
        model_name=args.model_name,
        containerWidth=args.containerWidth,
        scalingNumber=args.scalingNumber
    )
