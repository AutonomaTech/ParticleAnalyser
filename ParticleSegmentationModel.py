import os.path

import pyDeepP2SA as dp
import cv2
import logger_config
import json
from datetime import datetime

logger = logger_config.get_logger(__name__)


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
        # default settings, can be overriden
        self.points_per_side = 150
        #points_per_batch=128
        self.pred_iou_thresh = 0.75
        self.stability_score_thresh = 0.93
        #stability_score_offset=0.8
        self.crop_n_layers = 2
        self.crop_n_points_downscale_factor = 2
        self.min_mask_region_area = 0.0
        self.box_nms_tresh = 0.85 
        #use_m2m=True,
        self.image = cv2.imread(self.image_path)
        
        self.psd_bins_data = None  # this is data for plotting
        self.psd_data = None
        self._bins = None

        #parameters for processing
        self.circularity_threshold = 0
        self.diameter_threshold = 0

        # check if the image path and sam check point path exists
        if not os.path.exists(self.image_path): raise Exception('Image path does not exist')
        if not os.path.exists(self.sam_checkpoint_path): raise Exception('Sam checkpoint path does not exist')


    @property
    def bins(self):
        return self._bins
    
    @bins.setter
    def bins(self, bins):
        self._bins = sorted(bins)

    def generate_mask(self):
        logger.info(
            "Generating masks - image: {}, scaling factor: {} um/px, sam_checkpoint: {}, points_per_side: {}, pred_iou_thresh: {}, stability_score_thresh: {}, crop_n_layers: {}, crop_n_points_downscale_factor: {}, min_mask_region_area: {}, box_nms_tresh: {}",
            self.image_path, self.scaling_factor, self.sam_checkpoint_path, self.points_per_side, self.pred_iou_thresh,
            self.stability_score_thresh, self.crop_n_layers, self.crop_n_points_downscale_factor,
            self.min_mask_region_area, self.box_nms_tresh)
        start_time = datetime.now()
        masks = dp.generate_masks(self.image, self.sam_checkpoint_path, points_per_side=self.points_per_side,
                                  pred_iou_thresh=self.pred_iou_thresh,
                                  stability_score_thresh=self.stability_score_thresh,
                                  crop_n_layers=self.crop_n_layers,
                                  crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                                  min_mask_region_area=self.min_mask_region_area,
                                  box_nms_tresh=self.box_nms_tresh)

        self.masks = masks
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info("Generating masks took: {}", execution_time)
        return masks

    def visualise_masks(self):
        if self.masks is None:
            self.generate_mask()
        dp.visualise_masks(self.image, self.masks)

    def save_masks_to_csv(self, filename):
        dp.save_masks_to_csv(self.masks, filename, self.scaling_factor,self.diameter_threshold)

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
            self.segments = dp.get_segments(self.masks, self.scaling_factor)
        dp.plot_psd(self.diameter_threshold, self.circularity_threshold, num_bins, self.segments)

    def get_psd_bins(self, display=False):
        if self.segments is None:
            self.segments = dp.get_segments(self.masks, self.scaling_factor)
        self.psd_bins_data = dp.get_psd_data(self.diameter_threshold, self.circularity_threshold, self.bins, self.segments)

    def plot_psd_bins(self, display=False):
        if self.segments is None:
            self.segments = dp.get_segments(self.masks, self.scaling_factor)
        dp.plot_psd_bins(self.diameter_threshold, self.circularity_threshold, self.bins, self.segments)
        
    def setdiameter_threshold(self,diameter_threshold):
        self.diameter_threshold=diameter_threshold
    
    def setcircularity_threshold(self,circularity_threshold):
      self.circularity_threshold=circularity_threshold

    def get_psd_data(self):
        if self.segments is None:
              self.segments = dp.get_segments(self.masks, self.scaling_factor)
        psd_data = dp.get_psd_data(self.diameter_threshold, self.circularity_threshold, self.bins, self.segments)
        self.psd_data = {'differential': list(zip(tuple(self.bins), tuple(psd_data[1]))), 'cumulative':list(zip(tuple(self.bins), tuple(psd_data[2][::-1])))}
        return self.psd_data

    def get_totalArea(self):
      if self.segments is None:
          self.segments = dp.get_segments(self.masks, self.scaling_factor)
      return dp.calculate_totalArea(self.diameter_threshold, self.circularity_threshold,self.segments)


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
            self.segments = dp.get_segments(self.masks, self.scaling_factor)

        dp.save_psd(self.diameter_threshold, self.circularity_threshold, self.bins, self.segments, filename)

    def save_segments(self,filename):
        if self.segments is None:
            self.segments = dp.get_segments(self.masks, self.scaling_factor)
        with open(filename, 'w') as file:
            json.dump(self.segments, file)

    def open_segments(self):
        # Open the file in read mode
        with open('segments.txt', 'r') as file:
            # Read all lines from the file and strip newline characters
            self.segments = json.load(file)

    def save_psd_as_csv(self, id, directory):
        if self.psd_data is None:
            logger.error("No PSD data to export!")
            return
        
        # get values of the distrubutions
        cumulative = [i[1] for i in self.psd_data['cumulative']]
        differential = [i[1] for i in self.psd_data['differential']]

        dp.save_psd_as_csv(id, self.bins, cumulative, differential, directory)

