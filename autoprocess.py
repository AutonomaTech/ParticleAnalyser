import os
import subprocess
import argparse


def check_and_run_command(samples_root_folder):
    # Traverse through each main sample folder
    for folder in os.listdir(samples_root_folder):
        sample_folder = os.path.join(samples_root_folder, folder)

        if os.path.isdir(sample_folder):
            # Traverse each image sub-folder inside the sample folder
            for subfolder in os.listdir(sample_folder):
                image_folder_path = os.path.join(sample_folder, subfolder)

                if os.path.isdir(image_folder_path):
                    # Check for files ending with '_distribution.txt'
                    distribution_files = [f for f in os.listdir(image_folder_path) if f.endswith('_distribution.txt')]

                    if not distribution_files:
                        # If no distribution file is found, run the command
                        command = f"python startup.py {image_folder_path} 180000"
                        print(f"Running command: {command}")
                        subprocess.run(command, shell=True)


def main(samples_root_folder):
    """
    Main function to process all samples within the given root folder.
    """
    print(f"Processing all samples within the folder: {samples_root_folder}")
    check_and_run_command(samples_root_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process samples and run commands based on specific conditions.")
    parser.add_argument('samples_root_folder', type=str, help='Root folder path containing all sample subfolders.')

    args = parser.parse_args()

    main(args.samples_root_folder)
