import json
from datetime import datetime
import os
import sys
import ImageAnalysisModel as pa
import requests
# Load the parameters config json file
with open('parameters_config.json', 'r') as f:
    config = json.load(f)
current_path = os.getcwd()
parameter_sets = config['parameter_sets']

image_folder_path = os.path.join(current_path, "Samples/Circle")
checkpoint_folder = 'checkpoints'
model_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'
model_name = 'sam2.1_hiera_large.pt'
# in um
containerWidth = 180000
def download_model(checkpoint_folder, file_url, file_name):
    """
    Check sam model exists or not ,if not downloaded
    """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)
        print(f"Checkpoints folder created: {checkpoint_folder}")

    file_path = os.path.join(checkpoint_folder, file_name)
    if not os.path.exists(file_path):
        try:
            print(f"Downloading model file: {file_name}...")
            response = requests.get(file_url, stream=True)
            response.raise_for_status()  # check whether the request is successful or not
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Sam model downloaded successfully")
        except Exception as e:
            print(f"Error occurred during sam model downloading : {e}")
            sys.exit(1)
    else:
        print("Checkpoint already existed ，skip downloading")
download_model(checkpoint_folder, model_url, model_name)
for idx, params in enumerate(parameter_sets):
    # Subfolder for each parameter set
    parameter_folder_name = f"parameter_{idx + 1}"
    # Create Analyser instance
    analyser = pa.ImageAnalysisModel(image_folder_path, containerWidth)


    # startTime
    start_time = datetime.now()
    print(f"Analyzing particles {idx+1}/{len(parameter_sets)}，Start Time：{start_time}")

    # utilize parameters to analyze particles
    analyser.analyseValidationParticles(checkpoint_folder, testing_parameters=params)
    analyser.saveSegments()

    bins = [50, 101, 151, 201, 250, 300, 350, 400, 450, 500]
    analyser.saveResultsForValidation(bins,parameter_folder_name)
    analyser.formatResults()

    # End Time
    end_time = datetime.now()
    print(f"Complete particel analyzing: {idx+1}/{len(parameter_sets)}，End Time：{end_time}")

