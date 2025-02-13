# tester_flow.py
import pandas as pd
import json
from datetime import datetime
import os
import sys
import argparse
import requests
import logging
# Importing the ImageAnalysisModel from the analyser_module package
import ImageAnalysisModel as pa

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

def size_validate(generated_csv_path, manual_xlsx_path, distribution_path,output_csv_path):
    # Load CSV and Excel files

    generated_df = pd.read_csv(generated_csv_path)
    manual_df = pd.read_excel(manual_xlsx_path)

    # Calculate total area from both the generated CSV and manual Excel
    total_area_generated = generated_df['area'].sum()
    total_area_manual = manual_df['calculated area'].sum()
    generated_df.columns = generated_df.columns.str.strip()
    manual_df.columns = manual_df.columns.str.strip()
    # Calculate the total area difference and its percentage
    area_difference = abs(total_area_generated - total_area_manual)
    area_percentage_difference = (area_difference / total_area_manual) * 100 if total_area_manual != 0 else 0

    # Sort the generated data by diameter in descending order
    generated_df.sort_values(by='diameter', ascending=False, inplace=True)
    manual_df.sort_values(by='diameter', ascending=False, inplace=True)

    # Reset index after sorting to ensure alignment
    generated_df.reset_index(drop=True, inplace=True)
    manual_df.reset_index(drop=True, inplace=True)

    # Calculate differences and convert to percentages for diameters
    differences = abs(manual_df['diameter'] - generated_df['diameter'])
    percentages = (differences / manual_df['diameter']) * 100

    # Assign the percentage differences to the dataframe
    generated_df['diameter_difference'] = percentages.round(4).astype(str) + '%'

    # Append total area information to the DataFrame as new rows
    area_row = pd.DataFrame({
        'area': ['Total Area'],
        'perimeter': [total_area_generated],
        'diameter': [None],
        'circularity': [None],
        'diameter_difference': [None]
    })

    error_row = pd.DataFrame({
        'area': ['Error %'],
        'perimeter':[f"{area_percentage_difference:.2f}%"],
        'diameter': [None],
        'circularity': [None],
        'diameter_difference': [None]
    })

    generated_df = pd.concat([generated_df, area_row, error_row], ignore_index=True)

    input_string = ''
    try:
        # Take string in the psd file
        with open(distribution_path, 'r') as file:

            for line in file:
                input_string = line
        # Split the input string by comma
        elements = input_string.split(',')
        print(elements)

        # Filter the bins from the input; these are assumed to be the values before "% Passing"
        bin_array = elements[1:elements.index('Bottom') + 1]
        rows = len(bin_array)

        # Find the indices for passing and retaining percentages
        passing_start = elements.index('% Passing') + 1
        passing_end = elements.index('% Retained')
        retaining_start = elements.index('% Retained') + 1

        # Extract the passing and retaining percentages--only  values before bottom will be produced
        passing_data = elements[passing_start:passing_end]
        retaining_data = elements[retaining_start:]
        format_string = '.{}f'.format(4)
        passing = [format(max(float(num), 0), format_string) for num in passing_data]
        retaining = [format(max(float(num), 0), format_string) for num in retaining_data]
        generated_df['PSD'] = pd.Series(retaining[:len(generated_df)])
        generated_df['Cumulative Passing'] = pd.Series(passing[:len(generated_df)])
    except Exception as e:
        print("Distribution file can not be parsed:{} ", e)


    # Save the modified dataframe to a new CSV file
    generated_df.to_csv(output_csv_path, index=False)
    print(f"Validation CSV saved to: {output_csv_path}")
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


def main(image_folder_path, config_path, checkpoint_folder, model_url, model_name, containerWidth, scalingNumber,preCalculate_file_path,start):
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
        preCalculate_file_path (str): Scaling number used for scaling factor calculation.
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
            parameter_folder_path = os.path.join(image_folder_path, parameter_folder_name)
            if not start:
                if os.path.exists(parameter_folder_path):
                    logging.info(
                        f"Parameter folder '{parameter_folder_name}' already exists. Skipping this parameter set.")
                    continue
            else:
                logging.info(
                    f"Start is True. Processing parameter set '{parameter_folder_name}' regardless of existing folders.")
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
            output_csv = os.path.join(analyser.folder_path, f"{analyser.sampleID}_validating_report.csv")
            generated_csv = os.path.join(analyser.folder_path, f"{analyser.sampleID}.csv")
            distribution_path = os.path.join(analyser.folder_path, f"{analyser.sampleID}_distribution.txt")
            size_validate(generated_csv,preCalculate_file_path,distribution_path,output_csv)
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

    parser.add_argument(
        '--preCalculate_file_path',
        type=str,
        default=r'C:\Users\LiCui\Desktop\Samples\Circle_For_Validation\manual_calculation.xlsx',
        help='Pre manual calculation file .'
    )
    # New 'Start' parameter
    parser.add_argument(
        '--start',
        type=str2bool,
        nargs='?',
        default=False,
        help='If set to True, the process runs from the very first parameter_set. If False, it skips existing parameter folders.'
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
        scalingNumber=args.scalingNumber,
        preCalculate_file_path=args.preCalculate_file_path,
        start=args.start
    )
