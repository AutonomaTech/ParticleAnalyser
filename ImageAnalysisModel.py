import ContainerScalerModel as cs
import sizeAnalysisModel as sa
import ImageProcessingModel as ip
import CalibrationModel as cb
import logger_config
import ParticleSegmentationModel as psa
logger = logger_config.get_logger(__name__)
import os
import re
import csv
import configparser
import requests

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# -----------------------------------------------------------------------------
# ImageAnalysisModel Class
# -----------------------------------------------------------------------------
# The ImageAnalysisModel class serves as the central interface for analyzing
# images and extracting particle-related data. It integrates multiple models
# and utilities for preprocessing, scaling, segmentation, and analysis.
#
# Core Features:
# 1. Initialization:
#    - Takes the path to a folder containing images and a container width.
#    - Automatically sets up the sample ID, image processor, and scaling factor.
#
# 2. Preprocessing:
#    - Allows cropping and lighting adjustments to improve image quality.
#    - Supports overlaying images for enhanced visibility and size reduction.
#
# 3. Particle Segmentation and Analysis:
#    - Uses a ParticleSegmentationModel to segment particles and generate masks.
#    - Supports loading pretrained checkpoints for segmentation models.
#    - Analyzes particle size distribution (PSD) and saves data in various formats.
#
# 4. Results Management:
#    - Saves results such as particle masks, PSD data, and formatted XML files.
#    - Provides functionality to load pre-segmented data for analysis.
#
# 5. Visualization:
#    - Displays images and generated masks for inspection.
#
# 6. Customization:
#    - Supports adjustable bins and diameter thresholds for segmentation.
#
# Designed for extensibility and efficient image analysis, this class is
# structured to integrate with other models and tools seamlessly.
# -----------------------------------------------------------------------------


class ImageAnalysisModel:
    def __init__(self, image_folder_path, scalingNumber=None,containerWidth=None, sampleID=None,config_path=None):
        """
        Initializes the ImageAnalysisModel with an image folder path and container width. 
        Sets up the sample ID, image processor, and container scaler.

        Inputs:
        - image_folder_path: Path to the folder containing images for analysis.
        - containerWidth: Width of the container used for scaling.

        Output: None
        """
        self.calibration_file_path = "calibration.ini"
        self.calibratedSizeBin = None
        self.calibratedAreaBin = None
        self.sampleID = sampleID if sampleID else os.path.basename(
            image_folder_path)
        self.imageProcessor = ip.ImageProcessingModel(
            image_folder_path, self.sampleID)
        self.imagePath = self.imageProcessor.getImagePath()
        self.meshingImageFolderPath=None
        self.Scaler = cs.ContainerScalerModel(containerWidth)
        self.Scaler.updateScalingFactor(
            imageWidth=self.imageProcessor.getWidth(),scalingNumber=scalingNumber, containerWidth=containerWidth)
        self.diameter_threshold = 100000  # 10cm
        self.folder_path = image_folder_path
        self.meshingTotalSeconds=0
        self.totalSeconds=0
        self.analysisTime = 0
        self.numberofBins = 0
        self.p = None
        self.cb=None
        self.csv_filename = ""
        self.minimumArea=0
        self.totArea=0
        self.meshingSegmentAreas={}
        self.miniParticles=[]
        self.particles=[]
        self.csv_filename=""
        self.bins=None
        self.processImageOnly = False
        self.temperature = 3000
        self.crop_top = 0
        self.crop_left = 0
        self.crop_height = 0
        self.crop_width = 0
        self.mini_width = 100
        self.mini_height = 100
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.checkpoint_folder = 'checkpoints'
        self.normal_bins = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        self.model_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'
        self.model_name = 'sam2.1_hiera_large.pt'
        self.load_config()

    def load_config(self):
        # Load configuration file
        self.config.read(self.config_path)
        self.calculated_reminder_area = int(self.config.get('switch', 'CalculatedAdjustedBins_Area', fallback='0'))
        self.calculated_size = int(self.config.get('switch', 'CalculatedAdjustedBins_Size', fallback='0'))
        self.calculated_area = int(self.config.get('switch', 'CalculatedAdjustedBins_Area', fallback='0'))
        self.target_distribution = eval(self.config.get('PSD', 'lab', fallback='[]'))
        self.UseCalibratedBin = int(self.config.get('switch', 'UseCalibratedBin', fallback='0'))
        self.processImageOnly = self.str_to_bool(self.config.get('Image', 'processImageOnly', fallback='false'))
        self.temperature = int(self.config.get('Color', 'temperature', fallback='0'))
        industry_bins_string = self.config['analysis']['industryBin']
        self.industry_bins = self.parse_bins(industry_bins_string)
        if self.UseCalibratedBin != 0:
            self.load_calibrated_bins()
            # Load cropping parameters
        self.crop_top = int(self.config.get('Crop', 'Top', fallback='0'))
        self.crop_left = int(self.config.get('Crop', 'Left', fallback='0'))
        self.crop_height = int(self.config.get('Crop', 'Height', fallback='0'))
        self.crop_width = int(self.config.get('Crop', 'Width', fallback='0'))
        self.mini_width = int(self.config.get('Crop', 'minimumWidth', fallback='100'))
        self.mini_height = int(self.config.get('Crop', 'minimumHeight', fallback='100'))
    def load_calibrated_bins(self):
        calibration_config = configparser.ConfigParser()
        calibration_config.read(self.calibration_file_path)
        bin_key = str(self.UseCalibratedBin)
        # Read byArea and bySize from the calibration file
        self.calibratedAreaBin = eval(calibration_config.get(bin_key, 'byArea', fallback='[]'))
        self.calibratedSizeBin = eval(calibration_config.get(bin_key, 'bySize', fallback='[]'))
    def str_to_bool(self, s):
        if s.lower() in ['true', '1', 'yes']:
            return True
        return False
    def parse_bins(self, industry_bins_string):
        # Remove non-numeric characters and split by commas
        cleaned_string = re.sub(r'[\[\] ]', '', industry_bins_string)
        bins_list = cleaned_string.split(',')
        try:
            return [int(x) for x in bins_list]
        except ValueError as e:
            print("Error converting to integer:", e)
            return []

    def download_model(self):
        # Check if model exists; if not, download it
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder, exist_ok=True)
            print(f"Checkpoints folder created: {self.checkpoint_folder}")

        file_path = os.path.join(self.checkpoint_folder, self.model_name)
        if not os.path.exists(file_path):
            try:
                print(f"Downloading model file: {self.model_name}...")
                response = requests.get(self.model_url, stream=True)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print("Model downloaded successfully")
            except Exception as e:
                print(f"Error occurred during model downloading: {e}")
                raise
        else:
            print("Model already exists, skipping download.")

    def run_analysis(self):
        # Step 1: Download model
        self.download_model()

        # Step 2: Perform image processing
        self.crop_image()
        self.color_correction()
        self.evenLighting()
        self.overlayImage()
        if self.processImageOnly:
            logger.info("For this run process image only without analysis")
            return
        self.analyseParticles(self.checkpoint_folder, False)
        self.saveSegments()

        # Step 3:  Perform Image analysis
        if self.calibratedAreaBin:

            self.setBins(self.calibratedAreaBin)
        else:
            self.setBins(self.industry_bins if self.industry_bins else [38, 106, 1000, 8000])
        self.savePsdData()

        if self.calculated_reminder_area == 1:
            self.loadCalibrator()
            self.calculate_unsegmented_area()
            self.calibrated_bins_with_unSegementedArea()
            self.refactor_psd()
            distribution_fileName = os.path.join(self.folder_path, f'{self.sampleID}_refactored_distribution.txt')
            self.formatResults(byArea=True, distribution_filename=distribution_fileName)
        else:
            self.saveDistributionPlot()
            self.formatResults(byArea=True)

        if self.calibratedSizeBin:

            self.setBins(self.calibratedSizeBin)
        else:
            self.setBins(self.industry_bins if self.industry_bins else [38, 106, 1000, 8000])
        self.savePsdDataWithDiameter()
        self.formatResults(bySize=True)
        self.saveDistributionPlotForDiameter()
        self.saveResultsForNormalBinsOnly(self.normal_bins)
        self.formatResultsForNormalDistribution(True)

        if self.target_distribution:
            if self.calculated_size == 1:
                print("Calculating bins by size...")
                self.calibrate_bin_with_size(self.target_distribution)
            if self.calculated_area == 1:
                print("Calculating bins by area...")
                self.calibrate_bin_with_area(self.target_distribution)
        else:
            print("No target distribution provided. Skipping advanced bin calculations.")

    def analysewithCV2(self):
        self.csv_filename = os.path.join(
            self.folder_path, f"{self.sampleID}.csv")
        self.p.generate_with_cv2(self.csv_filename)

    def showImage(self):
        """
        Displays the processed image using the ImageProcessingModel.

        Input: None
        Output: Shows the image.
        """
        self.imageProcessor.showImage()

    def showMasks(self):
        """
        Displays the masks generated by the ParticleSegmentationModel, if available.

        Input: None
        Output: Shows mask visualization.
        """
        file_name = f"{self.folder_path}/{self.sampleID}_mask.png"
        self.p.visualise_masks(file_name)

    def setBins(self, bins):
        """
        Sets the number of bins in the ParticleSegmentationModel based on input.

        Inputs:
        - bins: List of bin boundaries.

        Output: None
        """
        self.bins=bins[:]
        if self.p is not None:
            self.numberofBins = len(bins)
            self.p.bins = bins[:]

    def loadModel(self, checkpoint_folder):
        """
        Loads the ParticleSegmentationModel with a specified checkpoint.

        Input:
        - checkpoint_folder: Path to the folder containing model checkpoint.

        Output: None
        """
        def loadSamModel(checkpoint_folder):
            os.makedirs(checkpoint_folder, exist_ok=True)
            checkpoint_filename = "sam2.1_hiera_large.pt"  # SAM2
            CHECKPOINT_PATH = os.path.join(
                checkpoint_folder, checkpoint_filename)
            return CHECKPOINT_PATH

        CHECKPOINT_PATH = loadSamModel(checkpoint_folder)
        self.p = psa.ParticleSegmentationModel(
            self.imagePath, CHECKPOINT_PATH, self.Scaler.scalingFactor)

    def analyseParticles(self, checkpoint_folder, testing):
        """
        Analyzes particles in the image by generating masks using the model, and calculates analysis time.

        Inputs:
        - checkpoint_folder: Path to the model checkpoint.
        - testing: Boolean flag to enable test mode.

        Output: None
        """
        def calculateAnalysisTime(duration):
            total_seconds = duration.total_seconds()
            # Total seconds for the first calculation (Entire image)
            # self.totalSeconds=total_seconds
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            self.analysisTime = f"PT{minutes}M{seconds:.1f}S"

        self.loadModel(checkpoint_folder)
        if testing:
            self.p.testing_generate_mask()
        else:
            self.p.generate_mask()

        calculateAnalysisTime(self.p.getExecutionTime())
        self.p.setdiameter_threshold(self.diameter_threshold)
        self.csv_filename = os.path.join(
            self.folder_path, f"{self.sampleID}.csv")
        self.p.save_masks_to_csv(self.csv_filename)
        self.showMasks()
    def analyseValidationParticles(self, checkpoint_folder,parameter_folder_name, testing_parameters=None):
        """
        Analyzes particles in the image by generating masks using the model, and calculates analysis time.

        Inputs:
        - checkpoint_folder: Path to the model checkpoint.
        - testing: Boolean flag to enable test mode.

        Output: None
        """
        def calculateAnalysisTime(duration):
            total_seconds = duration.total_seconds()
            # Total seconds for the first calculation (Entire image)
            # self.totalSeconds=total_seconds
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            self.analysisTime = f"PT{minutes}M{seconds:.1f}S"

        self.loadModel(checkpoint_folder)

        self.p.testing_generate_mask_1(**testing_parameters)


        calculateAnalysisTime(self.p.getExecutionTime())
        self.p.setdiameter_threshold(self.diameter_threshold)
        # Get Image folder Path
        original_folder_path = self.imageProcessor.getImageFolder()
        # Create subfolder for the current parameter set
        self.folder_path = os.path.join(original_folder_path, parameter_folder_name)
        os.makedirs(self.folder_path, exist_ok=True)
        self.csv_filename = os.path.join(
            self.folder_path, f"{self.sampleID}.csv")
        self.p.save_masks_to_csv(self.csv_filename)
        self.showMasks()

    def savePsdData(self):
        """
        Saves particle size distribution (PSD) data to a text file.

        Input: None
        Output: Saves PSD data to a TXT file.
        """
        self.p.get_psd_data()
        self.distributions_filename = os.path.join(
            self.folder_path, f"{self.sampleID}_byArea_distribution.txt")
        self.p.save_psd_as_txt(self.sampleID, self.distributions_filename)
        print(f"--> PSD data saved as TXT file: {self.distributions_filename}")
    def savePsdDataWithDiameter(self):
        """
        Saves particle size distribution (PSD) data to a text file.

        Input: None
        Output: Saves PSD data to a TXT file.
        """
        self.p.get_psd_data_with_diameter()
        self.distributions_filename = os.path.join(
            self.folder_path, f"{self.sampleID}_bySize_distribution.txt")
        self.p.save_psd_as_txt(self.sampleID, self.distributions_filename)
        print(f"--> PSD data saved as TXT file: {self.distributions_filename}")

    def savePsdDataForNormalBins(self):
        """
        Saves particle size distribution (PSD) data to a text file.

        Input: None
        Output: Saves PSD data to a TXT file.
        """
        self.p.get_psd_data()
        # self.distributions_filename = os.path.join(
        #     self.folder_path, f"{self.sampleID}_normalBin_distribution.txt")
        self.p.save_psd_as_txt_normal(self.sampleID, self.folder_path)
        # print(f"--> PSD data saved as TXT file: {self.distributions_filename}")
    def saveDistributionPlot(self):
        """
        Saves particle size distribution (PSD) data to a text file.

        Input: None
        Output: Saves PSD data to a TXT file.
        """

        self.p.plotBins(self.folder_path,self.sampleID)

    def saveDistributionPlotForDiameter(self):
        """
        Saves particle size distribution (PSD) data to a text file.

        Input: None
        Output: Saves PSD data to a TXT file.
        """

        self.p.plotBinsForDiameter(self.folder_path, self.sampleID)
    def saveDistributionPlotForNormalBins(self):
        """
        Saves particle size distribution (PSD) data to a text file.

        Input: None
        Output: Saves PSD data to a TXT file.
        """
        self.p.plotNormalBins(self.folder_path, self.sampleID)



    def saveResults(self, bins):
        """
        Saves particle segmentation results to CSV and distribution files after setting bins.

        Input:
        - bins: List of bin boundaries for the segmentation model.

        Output: Saves results to CSV and distribution files.
        """
        self.setBins(bins)
        if self.imageProcessor is None:
            raise ValueError("Image is not initialised")

        self.folder_path = self.imageProcessor.getImageFolder()
        self.csv_filename = os.path.join(
            self.folder_path, f"{self.sampleID}.csv")
        self.p.setdiameter_threshold(self.diameter_threshold)
        self.p.save_masks_to_csv(self.csv_filename)
        print(f"--> Masks saved to CSV file: {self.csv_filename}")

        self.savePsdData()
        self.saveDistributionPlot()

    def saveResultsForValidation(self, bins, parameter_folder_name):
        """
        Saves particle segmentation results to CSV and distribution files after setting bins.

        Input:
        - bins: List of bin boundaries for the segmentation model.
        - parameter_folder_name: Name of the subfolder for the current parameter set.

        Output: Saves results to CSV and distribution files in the specified subfolder.
        """
        self.setBins(bins)
        if self.imageProcessor is None:
            raise ValueError("Image is not initialised")



        # Generate new csv
        self.csv_filename = os.path.join(self.folder_path, f"{self.sampleID}.csv")
        self.p.setdiameter_threshold(self.diameter_threshold)
        self.p.save_masks_to_csv(self.csv_filename)
        print(f"--> Masks saved to CSV file: {self.csv_filename}")

        self.savePsdData()
        self.saveDistributionPlot()
    def saveResultsForNormalBinsOnly(self, bins):
        """
        Saves particle segmentation results to CSV and distribution files after setting bins.

        Input:
        - bins: List of bin boundaries for the segmentation model.

        Output: Saves results to CSV and distribution files.
        """
        self.setBins(bins)
        self.savePsdDataForNormalBins()
        self.saveDistributionPlotForNormalBins()
    def generateMasksForMeshing(self, testing):
        """
        Analyzes particles in the image by generating masks using the model for each segmented image
        and saves the resulting segment files to a corresponding folder.

        Inputs:
        - testing: Boolean flag to enable test mode.

        Output: None
        """

        def calculateTotalSeconds(duration):
            total_seconds = duration.total_seconds()
            self.meshingTotalSeconds += total_seconds

        # Step 1: Assign `self.meshingImageFolderPath`
        meshing_folder_name = "meshingImage"
        self.meshingImageFolderPath = os.path.join(self.folder_path, meshing_folder_name)

        # Step 2: Check if `meshingImage` folder exists
        if not os.path.exists(self.meshingImageFolderPath):
            print(f"Error: Folder '{meshing_folder_name}' not found at {self.folder_path}")
            return

        print(f"Meshing image folder path: {self.meshingImageFolderPath}")

        # Step 3: Assign `self.meshingSegmentsFolder`
        meshing_segment_folder_name = "meshingSegments"
        self.meshingSegmentsFolder = os.path.join(self.folder_path, meshing_segment_folder_name)

        # Create `meshingSegments` folder if it doesn't exist
        os.makedirs(self.meshingSegmentsFolder, exist_ok=True)
        print(f"Meshing segments folder path: {self.meshingSegmentsFolder}")

        # Step 4: List all image files in `meshingImageFolderPath` with natural sorting
        def natural_key(file_name):
            match = re.search(r'\d+', file_name)
            return int(match.group()) if match else float('inf')

        image_files = [
            os.path.join(self.meshingImageFolderPath, file)
            for file in sorted(os.listdir(self.meshingImageFolderPath),key=natural_key)
            if file.endswith(".png")  # Filter for PNG images
        ]

        if not image_files:
            print(f"No images found in {self.meshingImageFolderPath}")
            return

        print(f"Found {len(image_files)} images in {self.meshingImageFolderPath}")

        # Step 5: Loop through each image and process using the model
        for index, image_path in enumerate(image_files, start=1):
            print(f"Processing image: {image_path}")

            # Update the model's image path directly
            self.p.update_image_path(image_path)

            # Generate masks
            if testing:
                self.p.testing_generate_mask()
            else:
                self.p.generate_mask()

            # Step 6: Save masks to a corresponding CSV file
            self.p.setdiameter_threshold(self.diameter_threshold)
            csv_filename = os.path.join(self.meshingSegmentsFolder, f"meshing_{index}.csv")
            self.p.save_masks_to_csv(csv_filename)
            print(f"Segment file saved as: {csv_filename}")
            self.showMasks()

            # Calculate execution time
            calculateTotalSeconds(self.p.getExecutionTime())

        print("Finished processing all images.")

    def setScalingFactor(self, scalingFactor):
        self.Scaler.setScalingFactor(scalingFactor)

    def formatResults(self,byArea=False,bySize=False,distribution_filename=None):
        """
        Formats and displays analysis results, and saves formatted results as XML.

        Input: None
        Output: Prints formatted results and saves them to an XML file.
        """
        self.totArea = self.p.get_totalArea()
        print("-----------------------------------------------")
        print("Sample ID:", self.sampleID)
        print(f"Total Area: {self.totArea} um2")
        print(f"Total Area: {self.totArea / 100_000_000} cm2")
        print(f"Scaling Factor: {self.Scaler.scalingFactor} um/pixels")
        print(f"Scaling Number: {self.Scaler.scalingNumber} pixels")
        self.intensity = self.imageProcessor.getIntensity()
        print("Intensity:", self.intensity)
        print("Scaling Stamp:", self.Scaler.scalingStamp)
        print("Analysis Time:", self.analysisTime)
        print(f"Diameter Threshold: {self.p.diameter_threshold} um")
        print(f"Circularity Threshold: {self.p.circularity_threshold} um")
        print("-----------------------------------------------")
        print(f"CSV file: {self.csv_filename}")
        if distribution_filename:
            formatter = sa.sizeAnalysisModel(self.sampleID, self.csv_filename, distribution_filename,
                                             self.totArea, self.Scaler.scalingNumber,
                                             self.Scaler.scalingFactor, self.Scaler.scalingStamp,
                                             self.intensity, self.analysisTime, self.p.diameter_threshold,
                                             self.p.circularity_threshold)
        else:
            formatter = sa.sizeAnalysisModel(self.sampleID, self.csv_filename, self.distributions_filename,
                                             self.totArea, self.Scaler.scalingNumber,
                                             self.Scaler.scalingFactor, self.Scaler.scalingStamp,
                                             self.intensity, self.analysisTime, self.p.diameter_threshold,
                                             self.p.circularity_threshold)
        formatter.save_xml(byArea=byArea,bySize=bySize)

    def formatResultsForNormalDistribution(self,normalFlag):
        """
        Formats and displays analysis results, and saves formatted results as XML.

        Input: None
        Output: Prints formatted results and saves them to an XML file.
        """
        self.totArea = self.p.get_totalArea()
        print("-----------------------------------------------")
        print("Sample ID:", self.sampleID)
        print(f"Total Area: {self.totArea} um2")
        print(f"Total Area: {self.totArea / 100_000_000} cm2")
        print(f"Scaling Factor: {self.Scaler.scalingFactor} um/pixels")
        print(f"Scaling Number: {self.Scaler.scalingNumber} pixels")
        self.intensity = self.imageProcessor.getIntensity()
        print("Intensity:", self.intensity)
        print("Scaling Stamp:", self.Scaler.scalingStamp)
        print("Analysis Time:", self.analysisTime)
        print(f"Diameter Threshold: {self.p.diameter_threshold} um")
        print(f"Circularity Threshold: {self.p.circularity_threshold} um")
        print("-----------------------------------------------")
        print(f"CSV file: {self.csv_filename}")
        normalBins_distributions_filename = os.path.join(
            self.folder_path, f"{self.sampleID}_normalBin_distribution.txt")
        formatter = sa.sizeAnalysisModel(self.sampleID, self.csv_filename, normalBins_distributions_filename,
                                         self.totArea, self.Scaler.scalingNumber,
                                         self.Scaler.scalingFactor, self.Scaler.scalingStamp,
                                         self.intensity, self.analysisTime, self.p.diameter_threshold,
                                         self.p.circularity_threshold)
        formatter.save_xml(normalFlag=normalFlag)

    def saveSegments(self):
        """
        Saves segment data as JSON for later use.

        Input: None
        Output: Saves segment data to JSON file.
        """
        self.p.setdiameter_threshold(self.diameter_threshold)
        self.json_filename = os.path.join(
            self.folder_path, f"{self.sampleID}_segments.txt")
        self.p.save_segments(self.json_filename)
        print(f"Saving segments in {self.json_filename}")


    def loadSegments(self, checkpoint_folder, bins):
        """
        Loads segments from a JSON file and saves them to CSV and distribution files, useful for non-GPU environments.

        Inputs:
        - checkpoint_folder: Path to the model checkpoint.
        - bins: List of bin boundaries for the segmentation model.

        Output: Saves segment data to CSV and distribution files.
        """
        try:
            self.setFolderPath()
            self.json_masks_filename = os.path.join(
                self.folder_path, f"{self.sampleID}_segments.txt")

            if not os.path.exists(self.json_masks_filename):
                raise FileNotFoundError(
                    f"The file {self.json_masks_filename} was not found.")

            self.loadModel(checkpoint_folder)
            self.csv_filename = os.path.join(
                self.folder_path, f"{self.sampleID}.csv")
            self.p.setdiameter_threshold(self.diameter_threshold)
            self.p.save_segments_as_csv(
                self.json_masks_filename, self.csv_filename)


        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")

    def setFolderPath(self):
        """
        Sets the folder path for saving results, based on the initialized image processor.

        Input: None
        Output: Sets self.folder_path based on image folder path.
        """
        if self.imageProcessor is not None:
            self.folder_path = self.imageProcessor.getImageFolder()
        else:
            raise ValueError(
                "Image not initialized. Please ensure that 'imageProcessor' is properly initialized.")

    def crop_image(self):
        self.imageProcessor.cropImage(self.crop_width, self.crop_height, self.crop_left, self.crop_top,
                                      self.mini_height, self.mini_width)
        self.imagePath = self.imageProcessor.getImagePath()
        self.Scaler.updateScalingFactor(self.imageProcessor.getWidth())

    def evenLighting(self):
        self.imageProcessor.even_out_lighting()
        self.imagePath = self.imageProcessor.getImagePath()
        self.Scaler.updateScalingFactor(self.imageProcessor.getWidth())

    def evenLightingWithValidation(self,parameter_folder_path):
        self.imageProcessor.even_out_lighting_validation(parameter_folder_path)
        # self.imagePath = self.imageProcessor.getImagePath()


    def overlayImage(self):
        """
        Calls the ImageProcessingModel's overlayImage function to overlay the same picture 10 times and 
        reducing the size of the image if it is bigger than 8MB

        Input: None
        Output: lighter PNG file and containing the same image overlayed 10 times
        """
        self.imageProcessor.overlayImage()
        self.imagePath = self.imageProcessor.getImagePath()
        self.Scaler.updateScalingFactor(self.imageProcessor.getWidth())
    def overlayImageWithValidation(self):
        """
        Calls the ImageProcessingModel's overlayImage function to overlay the same picture 10 times and
        reducing the size of the image if it is bigger than 8MB

        Input: None
        Output: lighter PNG file and containing the same image overlayed 10 times
        """
        self.imageProcessor.overlayImage()
        self.imagePath = self.imageProcessor.getImagePath()



    def  meshingImage(self):
        self.imageProcessor.processImageWithMeshing()


    def plotBins(self):
        self.p.plotBins()

    def getAnalysisTime(self):
        total_seconds = self.totalSeconds+self.meshingTotalSeconds
        # Total seconds for the first calculation (Entire image)
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        self.analysisTime = f"PT{minutes}M{seconds:.1f}S"
        print(f"""The final analysing time is {self.analysisTime}""")
    def getSmallestAreaForFinalImage(self):
        """
        This function counts the number of data rows in a given file, ignoring the header row.

        Args:
        file_path (str): The path to the file.

        Returns:
        int: The number of data rows in the file.
        """
        particles=[]
        try:
            with open(self.csv_filename, 'r') as file:
                next(file)
                for line in file:
                    if line.strip():  # remove white space
                        area, perimeter, diameter, circularity = map(float, line.strip().split(','))
                        item = {
                            "area": area,
                            "perimeter": perimeter,
                            "diameter": diameter,
                            "circularity": circularity
                        }
                        particles.append(item)
            if len(particles) == 0:
                logger.error("There is no particles for minimumArea(ImageAnalysisModel) to be processed")
                return

            areas = [particle['area'] for particle in particles]
            sorted_areas = sorted(areas)
            self.mimumArea = format(max(float(sorted_areas[0] / 1000000), 0), '.8f')
            print(f'Minimu Area(ImageAnalysisModel) of the entire image analysis is :{self.mimumArea}')
            logger.info("Minimu Area(ImageAnalysisModel) of the entire image analysis is : {}", self.mimumArea)

        except Exception as e :
                logger.error("The give csv  file can  not be parsed due to {} error ",e)

    def getMeshingSegmentByCompareAreas(self):
        """
        Processes all CSV files in `self.meshingSegmentsFolder` and extracts particle areas.
        Compares each area with self.minimumArea and stores particles with areas smaller than self.minimumArea.

        Args:
        None

        Returns:
        None
        """
        # Ensure self.minimumArea is initialized
        if not hasattr(self, 'minimumArea'):
            self.minimumArea = float('inf')  # Set a large initial value if not set


        # Check if the folder exists
        if not os.path.exists(self.meshingSegmentsFolder):
            logger.error(f"Meshing segments folder {self.meshingSegmentsFolder} does not exist.")
            return

        # Get all CSV files in the folder
        segment_files = [
            os.path.join(self.meshingSegmentsFolder, file)
            for file in os.listdir(self.meshingSegmentsFolder)
            if file.endswith('.csv')
        ]

        if not segment_files:
            logger.error(f"No CSV files found in {self.meshingSegmentsFolder}")
            return

        # Process each CSV file
        for segment_file in sorted(segment_files):
            particles = []
            try:
                with open(segment_file, 'r') as file:
                    next(file)  # Skip the header row
                    for line in file:
                        if line.strip():  # Remove white space
                            # Parse the row
                            area, perimeter, diameter, circularity = map(float, line.strip().split(','))
                            formatted_area = format(max(float(area / 1000000), 0), '.8f')
                            item = {
                                "area": area,  # Store the original area
                                "perimeter": perimeter,
                                "diameter": diameter,
                                "circularity": circularity
                            }
                            # Compare formatted_area and not the original area
                            if float(formatted_area) < float(self.minimumArea):
                                self.miniParticles.append(item)
                            particles.append(item)

                if not particles:
                    logger.warning(f"No particles found in file {segment_file}")
                    continue

                print(f"Processed file {segment_file}: {len(particles)} particles")
                logger.info(f"Processed file {segment_file}: {len(particles)} particles")

            except Exception as e:
                logger.error(f"Failed to process file {segment_file} due to error: {e}")

        if not self.miniParticles:
            logger.warning("No particles smaller than minimumArea were found.")
        else:
            logger.info(f"Found {len(self.miniParticles)} particles smaller than minimumArea.")

    def processingMiniParticles(self):
        """
        Check the `self.miniParticles` list and write its contents to a new CSV file along with the contents of another existing CSV file.

        Args:
        None

        Returns:
        None
        """

        final_csv_path = os.path.join(self.folder_path, f"final_{self.sampleID}.csv")
        original_csv_path = os.path.join(self.folder_path, f"{self.sampleID}.csv")

        # Check if miniParticles is not empty
        if self.miniParticles:
            with open(final_csv_path, 'w', newline='') as csvfile:
                fieldnames = ['area', 'perimeter', 'diameter', 'circularity']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write the header
                writer.writeheader()

                # Write miniParticles to the new CSV
                for particle in self.miniParticles:
                    writer.writerow(particle)

                # Check if the original CSV exists and append its contents
                if os.path.exists(original_csv_path):
                    with open(original_csv_path, 'r') as original_csvfile:
                        reader = csv.DictReader(original_csvfile)
                        for row in reader:
                            writer.writerow(row)
                else:
                    print(f"Original CSV file {original_csv_path} does not exist.")
        else:
            print("No mini particles to process.")

    def save_final_results(self,bins):
        self.setBins(bins)
        self.p.setdiameter_threshold(self.diameter_threshold)
        final_csv = os.path.join(self.folder_path, f"final_{self.sampleID}.csv")
        regular_csv = os.path.join(self.folder_path, f"{self.sampleID}.csv")

        # Determine which file exists and set the appropriate output txt filename
        if os.path.exists(final_csv):
            input_file = final_csv
            output_txt = os.path.join(self.folder_path, "final_mesh_segments.txt")
        elif os.path.exists(regular_csv):
            input_file = regular_csv
            output_txt = os.path.join(self.folder_path, "final_segment.txt")
        else:
            print("No appropriate CSV file found.")
            return

        # Convert the CSV to TXT in JSON format
        self.convert_csv_to_json_txt(input_file, output_txt)

        # Assuming self.p has an open_segments method
        self.p.open_segments(output_txt)

        # Assuming a method to save PSD data
        self.savePsdData()

    def convert_csv_to_json_txt(self, csv_file_path, json_txt_output_path):
        """
        Converts a CSV file to a JSON-like format in plain text.

        Args:
        csv_file_path (str): Path to the input CSV file.
        json_txt_output_path (str): Path to the output text file with JSON format.

        Returns:
        None
        """
        try:
            data_list = []
            with open(csv_file_path, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # build String
                    line = "{\n" + ",\n".join(f"    {k}: {v}" for k, v in row.items()) + "\n},"
                    data_list.append(line)

            with open(json_txt_output_path, 'w') as output_file:
                # write to the csv file
                output_file.write("[\n" + ",\n".join(data_list)[:-1] + "\n]")
            print(f"Data successfully written to {json_txt_output_path}")

        except Exception as e:
            print(f"An error occurred while converting CSV to TXT: {e}")


    "--------------------Calibration Logic-------------------------------------------------------------------------------------------------"

    def loadCalibrator(self):
        """
        Loads the CalibrationModel with necessary arguments.

        Input:
        - toArea: Total Area of all segmented particles
        - csv_filename: Path to the folder containing segment csv file.
        - folder_path: Path to the folder containing image.
        - sampleId: Sample ID.
        Output: None
        """

        self.cb = cb.CalibrationModel(
            totArea=self.totArea,csv_filename=self.csv_filename,folder_path=self.folder_path,sampleId=self.sampleID,bins=self.bins)
    def calibrate_bin_with_size(self, target_distribution=None):
        self.cb.calibrate_bin_with_size(target_distribution)

    def calibrate_bin_with_area(self, target_distribution=None):
        self.cb.calibrate_bin_with_area(target_distribution)

    def calculate_unsegmented_area(self):
        self.cb.calculate_unsegmented_area()

    def calibrated_bins_with_unSegementedArea(self):
        self.cb.calibrated_bins_with_unSegementedArea()
    def refactor_psd(self):
        self.cb.refactor_psd()

    def color_correction(self):
        """
        Calls the ImageProcessingModel's color correction function to color correct the image

        Input: None
        Output: color corrected image
        """
        self.imageProcessor.colorCorrection(self.temperature)
        self.imagePath = self.imageProcessor.getImagePath()
