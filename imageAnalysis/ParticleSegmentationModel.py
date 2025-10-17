import os.path

import pyDeepP2SA as dp
import cv2

import json
import numpy as np
import configparser
from PIL import Image
from datetime import datetime
import configparser
from logger_config import get_logger
from Config_Manager import get_sam_parameters_path
parameterPath = get_sam_parameters_path()
logger = get_logger("Particle Segmentation")
#parameterPath = os.path.abspath(os.path.join(os.getcwd(), "imageAnalysis","samParameters.ini"))
class ParticleSegmentationModel:
 
    """

    ParticleSegmentationAnalysis is a class that provides a high level interface to the pyDeepP2SA library. 
    It provides a simple way to generate masks, visualise masks, save masks to csv, save mask image, save masks as images, save masked regions, identify spheres, plot psd, get psd bins, plot psd bins, save psd, save segments, open segments, save psd as csv.

    To use this class:
        1. Create an instance of the class with the image path, sam checkpoint path and pixel to micron scaling factor. 
        2. Generate masks which will segment the image into masks.
        3. Segements are scaled to a micron value

        points_per_side=64, # increasing improves the detection of lots of small particles # exp increase in processing
        pred_iou_thresh=0.85, # reducing the Iou and stablitiy score accepts more segments
        stability_score_thresh=0.8, # no sig. influence system is very confident on particle detected `98-99`
        box_nms_thresh=0.2 # reducing this reduces the number of duplicates

        iou_scores (Union[torch.Tensor, tf.Tensor]) — List of IoU scores.
        original_size (Tuple[int,int]) — Size of the orginal image.
        cropped_box_image (np.array) — The cropped image.
        pred_iou_thresh (float, optional, defaults to 0.88) — The threshold for the iou scores.
        stability_score_thresh (float, optional, defaults to 0.95) — The threshold for the stability score.
        mask_threshold (float, optional, defaults to 0) — The threshold for the predicted masks.
        stability_score_offset (float, optional, defaults to 1) — The offset for the stability score used in the _compute_stability_score method.

        What checkpoint to use?
        The models are the same except for neural network size, 
        B stands for "base" and is the smallest, L is "large" and H is "huge". 
        The paper reports that the performance difference between L and H isn't much 
        and I would recommend L if your machine supports it. 
        However, B is lighter and not far behind in performance.
        https://github.com/facebookresearch/segment-anything/issues/273

    """

    def __init__(self, image_path, sam_checkpoint_path, pixel_to_micron):
        self.image_path = image_path
        self.sam_checkpoint_path = sam_checkpoint_path
        self.masks = None
        self.scaling_factor = pixel_to_micron
        self.segments = None
        self.load_config(parameterPath)

        self.openedImage = Image.open(image_path)
        self.image = np.array(self.openedImage.convert("RGB"))

        self.psd_bins_data = None  # this is data for plotting
        self.psd_data = None
        self._bins = None

        # parameters for processing
        self.circularity_threshold = 0
        self.diameter_threshold = 0

        self.execution_time = None

        # check if the image path and sam check point path exists
        if not os.path.exists(self.image_path):
            raise Exception('Image path does not exist')
        if not os.path.exists(self.sam_checkpoint_path):
            raise Exception('Sam checkpoint path does not exist')
    
    def load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        
        if 'samParameters' in config:
            params = config['samParameters']
            
            # Default values and constraints
            defaults = {
                "points_per_side": (150, 32, 256),
                "points_per_batch": (128, 32, 256),
                "pred_iou_thresh": (0.8, 0.5, 0.95),
                "stability_score_thresh": (0.92, 0.85, 0.98),
                "stability_score_offset": (0.8, 0.5, 1.0),
                "crop_n_layers": (1, 0, 3),
                "crop_n_points_downscale_factor": (3, 1, 5),
                "min_mask_region_area": (0.0, 0.0, 1000.0),
                "box_nms_tresh": (0.2, 0.1, 0.5),
            }
            
            for key, (default, min_val, max_val) in defaults.items():
                value = params.get(key, default)
                value = type(default)(value)  
                
                if not (min_val <= value <= max_val):
                    logger.warning(f"Invalid {key}: {value}. Setting to {default}.")
                    value = default
                
                setattr(self, key, value)

            self.use_m2m = params.get('use_m2m', 'True').lower() in ('true', '1', 'yes')
        else:
            logger.error("Sam Parameters could not be obtained from samParameters.ini")


    def load_image(self, image_path):
        """Load image from the specified path and update the image attribute."""
        if not os.path.exists(image_path):
            raise Exception('Image path does not exist')

        self.openedImage = Image.open(image_path)
        self.image = np.array(self.openedImage.convert("RGB"))

    def update_image_path(self, new_image_path):
        """Update image path and reload the image."""
        self.load_image(new_image_path)

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, bins):
        self._bins = sorted(bins)

    def getExecutionTime(self):
        if self.execution_time is not None:
            return self.execution_time

    def generate_mask(self):
        logger.info(
            "Generating masks - image: {}, scaling factor: {} um/px, sam_checkpoint: {}, points_per_side: {},points_per_batch: {}, pred_iou_thresh: {}, stability_score_thresh: {}, \
            stability_score_offset:{}, crop_n_layers: {}, crop_n_points_downscale_factor: {}, min_mask_region_area: {}, box_nms_tresh: {}, use_m2m: {}",
            self.image_path, self.scaling_factor, self.sam_checkpoint_path, self.points_per_side, self.points_per_batch, self.pred_iou_thresh,
            self.stability_score_thresh, self.stability_score_offset, self.crop_n_layers, self.crop_n_points_downscale_factor,
            self.min_mask_region_area, self.box_nms_tresh, self.use_m2m)
        start_time = datetime.now()

        masks = dp.generate_masks_updated(self.image, self.sam_checkpoint_path,
                                  points_per_side=self.points_per_side,
                                  points_per_batch=self.points_per_batch,
                                  pred_iou_thresh=self.pred_iou_thresh,
                                  stability_score_thresh=self.stability_score_thresh,
                                  stability_score_offset=self.stability_score_offset,
                                  crop_n_layers=self.crop_n_layers,
                                  crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                                  min_mask_region_area=self.min_mask_region_area,
                                  box_nms_tresh=self.box_nms_tresh,
                                  use_m2m=self.use_m2m)
        # self.setdiameter_threshold(10)
        # self.masks=dp.deleteWrongAreas(masks,self.scaling_factor,self.diameter_threshold)
        self.masks = masks
        end_time = datetime.now()
        self.execution_time = end_time - start_time
        logger.info("Generating masks took: {}", self.execution_time)
        return masks

    def testing_generate_mask(self):
        # function TO Do test opn Colab to speed up process of testing. The results are not accurate
        start_time = datetime.now()

        masks = dp.generate_masks(self.image,
                                  self.sam_checkpoint_path,
                                  points_per_side=4,
                                  points_per_batch=1,
                                  pred_iou_thresh=0.1,
                                  stability_score_thresh=0.1,
                                  stability_score_offset=0.1,
                                  crop_n_layers=1,
                                  crop_n_points_downscale_factor=2,
                                  min_mask_region_area=100,
                                  box_nms_tresh=0.5,
                                  use_m2m=False
                                  )
        self.masks = masks
        end_time = datetime.now()
        self.execution_time = end_time - start_time
        logger.info("Generating masks took: {}", self.execution_time)
        return masks

    def testing_generate_mask_1(self, pred_iou_thresh=None, stability_score_thresh=None, stability_score_offset=None,
                                crop_n_layers=None, crop_n_points_downscale_factor=None, min_mask_region_area=None,
                                box_nms_tresh=None, use_m2m=False):
        # For validation
        logger.info(
            "Generating masks for validation - image: {}, scaling factor: {} um/px,  points_per_side: {},points_per_batch: {}, pred_iou_thresh: {}, stability_score_thresh: {}, \
            stability_score_offset:{}, crop_n_layers: {}, crop_n_points_downscale_factor: {}, min_mask_region_area: {}, box_nms_tresh: {}, use_m2m: {}",
            self.image_path, self.scaling_factor,  self.points_per_side, self.points_per_batch,
            pred_iou_thresh,
            stability_score_thresh, stability_score_offset, crop_n_layers,
            crop_n_points_downscale_factor,
            min_mask_region_area, box_nms_tresh, use_m2m)
        start_time = datetime.now()
        masks = dp.generate_masks(
            self.image,
            self.sam_checkpoint_path,
            points_per_side=150,
            points_per_batch=128,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            box_nms_tresh=box_nms_tresh,
            use_m2m=use_m2m
        )
        self.masks = masks
        end_time = datetime.now()
        self.execution_time = end_time - start_time
        logger.info("Generating masks: {}", self.execution_time)
        return masks

    def visualise_masks(self, mask_file_name):
        if self.masks is None:
            self.generate_mask()

        dp.visualise_masks(self.image, self.masks, mask_file_name)

    def opposite_masks(self):
        if self.masks is None:
            self.generate_mask()
        dp.visualiseRemainingfromMasks(self.image, self.masks)
        dp.find_smallest_area_with_SAM

    def save_masks_to_csv(self, filename):
        if self.masks is None:
            logger.error("No mask to save!")
            return
        dp.save_masks_to_csv(self.masks, filename,
                             self.scaling_factor, self.diameter_threshold)

    def save_mask_image(self, filename):
        if self.masks is None:
            logger.error("No mask to save!")
            return
        dp.save_masks_image(self.image, self.masks, filename)

    def save_masks_as_images(self, filename):
        if self.masks is None:
            logger.error("No mask to save!")
            return
        dp.save_masks_as_images(self.image, self.masks, filename)

    def save_masked_regions(self, filename):
        if self.masks is None:
            logger.error("No mask to save!")
            return
        dp.save_masked_regions(self.image, self.masks, filename)

    def identify_spheres(self, display=False):
        if self.masks is None:
            logger.error("No mask to identify spheres!")
            return
        dp.plot_diameters(self.image, self.masks, self.diameter_threshold, self.circularity_threshold, self.scaling_factor,
                          display)

    def identify_annotate_spheres(self, display=False):
        if self.masks is None:
            logger.error("No mask to identify spheres!")
            return
        dp.ind_mask(self.image, self.masks, self.diameter_threshold, self.circularity_threshold, self.scaling_factor,
                    display)

    def plot_psd(self, num_bins, display=False):
        """
        Plot the particle size distribution with threasholds. Values must be higher than the thresholds to be considered.
        :param diameter_threshold:
        :param circularity_threshold:
        :param num_bins:
        :param display:
        :return:
        """
        if self.masks is None:
            logger.error("No mask to identify spheres!")
            return
        if self.segments is None:
            self.segments = dp.get_segments(
                self.masks, self.scaling_factor, self.diameter_threshold)
        dp.plot_psd(self.diameter_threshold,
                    self.circularity_threshold, num_bins, self.segments)

    def get_psd_bins(self, display=False):
        if self.segments is None:
            self.segments = dp.get_segments(
                self.masks, self.scaling_factor, self.diameter_threshold)
        self.psd_bins_data = dp.get_psd_data(
            self.diameter_threshold, self.circularity_threshold, self.bins, self.segments)

    def plot_psd_bins(self, display=False):
        if self.segments is None:
            self.segments = dp.get_segments(
                self.masks, self.scaling_factor, self.diameter_threshold)
        dp.plot_psd_bins(self.diameter_threshold,
                         self.circularity_threshold, self.bins, self.segments)

    def setdiameter_threshold(self, diameter_threshold):
        self.diameter_threshold = diameter_threshold

    def setcircularity_threshold(self, circularity_threshold):
        self.circularity_threshold = circularity_threshold
    # Based on Particles

    def get_psd_data(self):
        if self.segments is None:
            self.segments = dp.get_segments(
                self.masks, self.scaling_factor, self.diameter_threshold)
        extended_bins = self.bins + [float('inf')]
        psd_data = dp.get_psd_data(
            self.diameter_threshold, self.circularity_threshold, extended_bins, self.segments, False)
        self.psd_data = {'differential': list(zip(tuple([0]+self.bins), tuple(
            psd_data[1]))), 'cumulative': list(zip(tuple([0]+self.bins), tuple(psd_data[2][::-1])))}
        # print(self.psd_data)

        return self.psd_data

    def get_psd_data_with_diameter(self):
        if self.segments is None:
            self.segments = dp.get_segments(
                self.masks, self.scaling_factor, self.diameter_threshold)

        psd_data = dp.custom_psd_data1(self.diameter_threshold, self.circularity_threshold, self.bins, self.segments,
                                       reverse_cumulative=True)
        self.psd_data = {'differential': list(zip(tuple([0] + self.bins), tuple(
            psd_data[1]))), 'cumulative': list(zip(tuple([0] + self.bins), tuple(psd_data[2][::-1])))}
        return self.psd_data

    def get_totalArea(self):  # , withOverlappingArea):
        if self.segments is None:
            self.segments = dp.get_segments(
                self.masks, self.scaling_factor, self.diameter_threshold)
        # overlapping = 0
        # if not withOverlappingArea:
           # overlapping = dp.calculate_overlapping_area(
            # self.masks, self.scaling_factor)
        area = dp.calculate_totalArea(
            self.diameter_threshold, self.circularity_threshold, self.segments)
        return area  # -overlapping

    def save_psd(self, filename):
        """
        Save the particle size distribution with threasholds. Values must be higher than the thresholds to be considered.
        :param diameter_threshold:
        :param circularity_threshold:
        :param num_bins:
        :param filename:
        :return:
        """
        if self.masks is None:
            logger.error("No mask to identify spheres!")
            return
        if self.segments is None:
            self.segments = dp.get_segments(
                self.masks, self.scaling_factor, self.diameter_threshold)

        dp.save_psd(self.diameter_threshold, self.circularity_threshold,
                    self.bins, self.segments, filename)

    def save_segments(self, filename):
        if self.segments is None:
            self.segments = dp.get_segments(
                self.masks, self.scaling_factor, self.diameter_threshold)
        with open(filename, 'w') as file:
            json.dump(self.segments, file)

    def open_segments(self, filename):
        # Open the file in read mode
        with open(filename, 'r') as file:
            # Read all lines from the file and strip newline characters
            self.segments = json.load(file)

    def getSegments(self):
        if self.segments is not None:
            return self.segments

    def save_psd_as_txt(self, id, directory):
        if self.psd_data is None:
            logger.error("No PSD data to export!")
            return
        # get values of the distrubutions
        cumulative = [i[1] for i in self.psd_data['cumulative']]
        differential = [i[1] for i in self.psd_data['differential']]

        dp.save_psd_as_txt(id, self.bins, cumulative, differential, directory)

    def save_psd_as_txt_normal(self, id, directory):
        if self.psd_data is None:
            logger.error("No PSD data to export!")
            return
        # get values of the distrubutions
        cumulative = [i[1] for i in self.psd_data['cumulative']]
        differential = [i[1] for i in self.psd_data['differential']]

        dp.save_psd_as_txt_normal(
            id, self.bins, cumulative, differential, directory)

    def save_segments_as_csv(self, txt_filename, csv_filename):
        self.segments = dp.save_segments_as_csv(
            txt_filename, csv_filename, self.diameter_threshold)

    def generate_with_cv2(self, csv_filename):
        # if self.masks is None:
        # self.generate_mask()
        # leftoverImage=dp.visualiseRemainingfromMasks(self.image, self.masks)
        min_area_found = dp.find_smallest_area_with_SAM(csv_filename)
        print(min_area_found)
        dp.detect_rocks_withCV2(self.image, float(min_area_found))

    def plotBins(self, folder_path, sampleId):
        # dp.plot_psd_bins(self.diameter_threshold, self.circularity_threshold, self.bins, self.segments)
        fileName = f"{folder_path}/{sampleId}_area_plot.png"
        dp.plot_psd_bins2(self.diameter_threshold,
                          self.circularity_threshold, self.bins, self.segments, fileName)

    def plotBinsForDiameter(self, folder_path, sampleId):
        # dp.plot_psd_bins(self.diameter_threshold, self.circularity_threshold, self.bins, self.segments)
        fileName = f"{folder_path}/{sampleId}_size_plot.png"
        dp.plot_psd_bins4(self.diameter_threshold,
                          self.circularity_threshold, self.bins, self.segments, fileName)

    def plotNormalBins(self, folder_path, sampleId):
        # dp.plot_psd_bins(self.diameter_threshold, self.circularity_threshold, self.bins, self.segments)
        fileName = f"{folder_path}/{sampleId}_normalBin_plot.png"
        dp.plot_psd_bins2(self.diameter_threshold,
                          self.circularity_threshold, self.bins, self.segments, fileName)
