import csv
import os
import numpy as np
import json
from scipy.interpolate import make_interp_spline
from skimage import measure
import torch
import torchvision
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from skimage.segmentation import clear_border
import math
from scipy.signal import argrelextrema
from skimage import measure
from scipy.optimize import curve_fit
from PIL import Image
import matplotlib.patches as patches
from os import path
import gc
# JH added save_masks function 18082024
# JH modified plot_diameters function to include default display = False parameter.
#

# Particle shape and size analysis section

"""
points_per_side: This parameter likely controls the number of points sampled along each side of the object being segmented. It could affect the granularity or detail of the segmentation.

pred_iou_thresh: This parameter could represent the threshold for the predicted Intersection over Union (IoU) score used to determine whether a segment is considered valid. IoU is a measure of overlap between predicted and ground truth masks.

stability_score_thresh: This parameter might represent a threshold for a stability score used to assess the stability of the segmentation. It could be a measure of how consistent or reliable the segmentation results are.

crop_n_layers: This parameter may control the number of layers cropped from the segmentation model. Cropping layers can sometimes improve segmentation performance by focusing on relevant features.

crop_n_points_downscale_factor: This parameter may control how much the number of points sampled is downscaled after cropping layers from the model. It could be a way to adjust the level of detail in the segmentation.

min_mask_region_area: This parameter likely represents the minimum area threshold for a mask region to be considered valid. Regions below this threshold might be discarded as noise or artifacts.

"""
# Global cache for SAM2 model and generator
_cached_sam2_model = None
_cached_mask_generator = None
_cached_checkpoint_path = None


def generate_masks_updated(image, sam2_checkpoint,
                   points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh, stability_score_offset,
                   crop_n_layers, crop_n_points_downscale_factor, min_mask_region_area, box_nms_tresh, use_m2m):
    global _cached_sam2_model, _cached_mask_generator, _cached_checkpoint_path

    # Check if we need to load/reload the model
    if (_cached_sam2_model is None or _cached_checkpoint_path != sam2_checkpoint):

        # Clean up existing model if any
        if _cached_sam2_model is not None:
            print(f"Cleaning up existing SAM2 model...")
            del _cached_sam2_model
            del _cached_mask_generator
            _cached_sam2_model = None
            _cached_mask_generator = None
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Loading SAM2 model from checkpoint: {sam2_checkpoint}")

        # Original model loading code
        model_type = "vit_h"

        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on float32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        current_model_cfg = "sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
        model_cfg = os.path.abspath(current_model_cfg)

        # Build and cache the model
        _cached_sam2_model = build_sam2(model_cfg, sam2_checkpoint,
                                        device=device, apply_postprocessing=False)

        # Create and cache the mask generator
        _cached_mask_generator = SAM2AutomaticMaskGenerator(
            model=_cached_sam2_model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            box_nms_thresh=box_nms_tresh,
            use_m2m=use_m2m
        )

        # Update cached checkpoint path
        _cached_checkpoint_path = sam2_checkpoint
        print(f"SAM2 model loaded and cached successfully")

    else:
        print(f"Using cached SAM2 model for analysis")

    # Generate masks using cached generator
    masks = _cached_mask_generator.generate(image)
    return masks


def generate_masks(image, sam2_checkpoint,
                   points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh, stability_score_offset,
                   crop_n_layers, crop_n_points_downscale_factor, min_mask_region_area, box_nms_tresh, use_m2m):

    model_type = "vit_h"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)

    current_model_cfg = "sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
    model_cfg = os.path.abspath(current_model_cfg)
    sam2 = build_sam2(model_cfg, sam2_checkpoint,
                      device=device, apply_postprocessing=False)
    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        box_nms_thresh=box_nms_tresh,
        use_m2m=use_m2m
    )

    masks = mask_generator_2.generate(image)

    return masks


# def visualise_masks(image, masks, file_name):
#     def show_anns(anns, borders=True):
#         if len(anns) == 0:
#             return
#         sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
#         ax = plt.gca()
#         ax.set_autoscale_on(False)
#
#         img = np.ones((sorted_anns[0]['segmentation'].shape[0],
#                       sorted_anns[0]['segmentation'].shape[1], 4))
#         img[:, :, 3] = 0  # Set transparency to 0 (fully transparent)
#
#         for ann in sorted_anns:
#             m = ann['segmentation']
#             # Random color with 50% transparency
#             color_mask = np.concatenate([np.random.random(3), [0.5]])
#             img[m] = color_mask  # Apply mask with the color and alpha value
#
#             if borders:
#                 contours, _ = cv2.findContours(
#                     m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 # Try to smooth contours
#                 contours = [cv2.approxPolyDP(
#                     contour, epsilon=0.01, closed=True) for contour in contours]
#                 # Draw contour with transparency
#                 cv2.drawContours(img, contours, -1,
#                                  (0, 0, 1, 0.4), thickness=1)
#
#         ax.imshow(img)
#
#     # Plot the original image with masks overlaid
#     plt.figure(figsize=(20, 20))
#     plt.imshow(image)
#     show_anns(masks)
#     plt.axis('off')  # Hide axis
#     plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
#     # plt.show()
#     # Save the generated image
#
#     print(f"Mask saved to {file_name}")

##### Update visual_mask to make sure system force release resource and collect garbage
def visualise_masks(image, masks, file_name):
    import matplotlib
    matplotlib.use('Agg')  # Must be set before importing pyplot
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import gc

    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                       sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0  # Set transparency to 0 (fully transparent)

        for ann in sorted_anns:
            m = ann['segmentation']
            # Random color with 50% transparency
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask  # Apply mask with the color and alpha value

            if borders:
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Try to smooth contours
                contours = [cv2.approxPolyDP(
                    contour, epsilon=0.01, closed=True) for contour in contours]
                # Draw contour with transparency
                cv2.drawContours(img, contours, -1,
                                 (0, 0, 1, 0.4), thickness=1)

        ax.imshow(img)

    try:
        # Create figure
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')  # Hide axis

        # Reduce DPI to minimize memory usage
        plt.savefig(file_name, dpi=150, bbox_inches='tight', pad_inches=0)

        print(f"Mask saved to {file_name}")

    finally:
        # Ensure resources are released even if an error occurs
        plt.close('all')  # Close all figures
        gc.collect()  # Force garbage collection
def visualiseRemainingfromMasks(image, masks, background_color=(255, 255, 255)):
    """
    Visualizes the image with masked areas removed, leaving only unmasked regions visible.

    Args:
        image (numpy.ndarray): The original image as a NumPy array (H x W x C).
        masks (list): A list of masks where each mask is a 2D NumPy array (H x W).
        background_color (tuple): RGB color to replace masked areas (default is white).
    """
    # Create a combined mask from all individual masks
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    for mask in masks:
        # Assuming each mask is a dictionary with 'segmentation' key
        combined_mask |= mask['segmentation']

    # Convert the combined mask to a 3D boolean array (for RGB channels)
    combined_mask_3d = np.stack([combined_mask] * 3, axis=-1)

    # Replace the masked areas with the background color
    modified_image = np.copy(image)

    # Apply the background color to each channel (R, G, B) where the mask is True
    for i in range(3):  # Loop over the three channels: Red, Green, Blue
        modified_image[:, :, i][combined_mask] = background_color[i]

    return modified_image


def find_smallest_area_with_SAM(input_filename):
    try:
        # Read the CSV file into a list of dictionaries
        with open(input_filename, mode='r') as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)

        # Sort the rows by the 'area' field in ascending order (smallest first)
        sorted_rows = sorted(rows, key=lambda row: float(row['area']))

        # Get the smallest area (the first entry in the sorted list)
        smallest_area = sorted_rows[0]['area']

        # Write the sorted data back to the same CSV file
        with open(input_filename, mode='w', newline='') as outfile:
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_rows)

        print(f"Smallest Area: {smallest_area}")

    except Exception as e:
        print(f"An error occurred: {e}")

    return smallest_area


def detect_rocks_withCV2(image, max_area):
    """
    Detects potential rocks in the image by finding contours in the unmasked regions.

    Args:
        image (numpy.ndarray): The input image with background areas removed.
        min_area (int): Minimum area threshold for detected contours.
        max_area (int): Maximum area threshold for detected contours.

    Returns:
        image_with_contours (numpy.ndarray): Image with contours drawn on detected rocks.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to isolate potential rocks (adjust threshold value as needed)
    _, thresholded = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (min_area and max_area)
    filtered_contours = [
        contour for contour in contours if 0 < cv2.contourArea(contour) < max_area]

    # Create an empty image to draw contours
    image_with_contours = np.copy(image)

    # Draw contours on the image
    cv2.drawContours(image_with_contours, filtered_contours, -
                     1, (0, 255, 0), 2)  # Green contours
    print(len(filtered_contours))


def save_masks_image(image, masks, filename):
    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.35)))

    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.axis('off')
    # plt.title("Segmented image with " + str(len(masks)) + " segments" )
    # plt.show()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)


def save_masked_regions(image, masks, filename):
    # Convert the image to a numpy array
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = np.array(image_rgb)

    def save_individual_mask(mask, mask_id, filename):
        # Use the mask to index into the image and extract the region
        masked_region = image_array * np.expand_dims(mask, axis=-1)

        # Convert the masked region back to an image
        masked_region_image = Image.fromarray(masked_region.astype(np.uint8))

        # Save the masked region image
        mask_filename = f'{filename}_mask_{mask_id}.png'
        masked_region_image.save(mask_filename)

    # Loop through each mask and save it
    for mask_id, ann in enumerate(masks):
        mask = ann['segmentation']
        save_individual_mask(mask, mask_id, filename)


def save_masks_as_images(image, masks, filename):

    def save_individual_mask(mask, mask_id, filename):
        # Create a black background image
        height, width = mask.shape
        black_background = np.zeros((height, width), dtype=np.uint8)

        # Create the white mask
        white_mask = np.where(mask == 1, 255, black_background)

        # Save the mask image
        mask_filename = f'{filename}_mask_{mask_id}.png'
        plt.imsave(mask_filename, white_mask, cmap='gray')

    # Loop through each mask and save it
    for mask_id, ann in enumerate(masks):
        mask = ann['segmentation']
        save_individual_mask(mask, mask_id, filename)


def save_masks_to_csv(masks, csv_directory, pixel_to_micron, diameter_threshold):
    with open(csv_directory, 'w', newline='') as csvfile:
        fieldnames = ['area', 'perimeter', 'diameter', 'circularity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for mask in masks:
            # Extract the segmentation from the mask
            segmentation = mask['segmentation']

            # Create a labeled mask using the segmentation and the 'true' value
            labeled_mask = np.zeros_like(segmentation, dtype=np.uint8)
            labeled_mask[segmentation] = 1
            labeled_mask = measure.label(labeled_mask)

            # Remove segments touching the border
            cleared_mask = clear_border(labeled_mask)

            # Loop over each connected component and extract its region props
            for region in measure.regionprops(cleared_mask):
                # Convert area, perimeter, and diameter from pixels to micrometers
                area_pixels = region.area
                perimeter_pixels = region.perimeter
                diameter_pixels = region.major_axis_length

                # Convert measurements to microns
                area = area_pixels * (pixel_to_micron**2)
                perimeter = perimeter_pixels * pixel_to_micron
                diameter = diameter_pixels * pixel_to_micron

                # Calculate circularity as 4π(area/perimeter^2)
                circularity = (4 * np.pi * area) / (perimeter ** 2)

                if diameter == 0.0 and area > 0.0:
                    diameter = np.sqrt(area / np.pi) / 2

                if diameter < diameter_threshold:
                    writer.writerow({'area': area, 'perimeter': perimeter,
                                    'diameter': diameter, 'circularity': circularity})


def get_segments(masks, pixel_to_micron, diameter_threshold=0):

    segments = []
    for mask in masks:
        # Extract the segmentation from the mask
        segmentation = mask['segmentation']

        # Create a labeled mask using the segmentation and the 'true' value
        labeled_mask = np.zeros_like(segmentation, dtype=np.uint8)
        labeled_mask[segmentation] = 1
        labeled_mask = measure.label(labeled_mask)

        # Remove segments touching the border
        cleared_mask = clear_border(labeled_mask)

        # Loop over each connected component and extract its region props
        for region in measure.regionprops(cleared_mask):
            # Convert area, perimeter, and diameter from pixels to micrometers
            area_pixels = region.area
            perimeter_pixels = region.perimeter
            diameter_pixels = region.major_axis_length

            # Convert measurements to microns
            area = area_pixels * (pixel_to_micron ** 2)
            perimeter = perimeter_pixels * pixel_to_micron
            diameter = diameter_pixels * pixel_to_micron

            # Calculate circularity as 4π(area/perimeter^2)
            circularity = (4 * np.pi * area) / (perimeter ** 2)

            if diameter == 0.0 and area > 0.0:
                diameter = np.sqrt(area / np.pi) / 2

            if diameter_threshold != 0 and diameter < diameter_threshold:
                # Write region props to CSV file
                segments.append({'area': area, 'perimeter': perimeter,
                                'diameter': diameter, 'circularity': circularity})

    return segments


def calculate_overlapping_area(masks, pixel_to_micron):
    """
    Calculate the total overlapping area between masks.

    Parameters:
    - masks (list): A list of dictionaries from the SAM model. Each dictionary contains:
        - 'segmentation': A binary mask (2D NumPy array) representing a single object.
    - pixel_to_micron (float): Conversion factor from pixels to microns.

    Returns:
    - overlapping_area_micron (float): The total overlapping area in square microns.
    """
    if not masks:
        return 0

    # Initialize a combined mask
    combined_mask = np.zeros_like(masks[0]['segmentation'], dtype=np.uint8)

    # Combine all masks
    for mask in masks:
        if isinstance(mask, dict) and 'segmentation' in mask:
            # Ensure binary mask
            mask_data = (mask['segmentation'] > 0).astype(np.uint8)
            combined_mask += mask_data

    # Count overlapping pixels (values > 1)
    overlapping_area_pixels = np.sum(combined_mask > 1)

    # Convert to microns (area in square microns)
    overlapping_area_micron = overlapping_area_pixels * (pixel_to_micron ** 2)

    print(f"Overlapping area: {overlapping_area_micron:.2f} square microns")
    return overlapping_area_micron


def plot_diameters(image, masks, diameter_threshold, circularity_threshold, pixel_to_micron, display=False):
    fig, ax = plt.subplots()

    for mask in masks:
        # Extract the segmentation from the mask
        segmentation = mask['segmentation']

        # Create a labeled mask using the segmentation and the 'true' value
        labeled_mask = np.zeros_like(segmentation, dtype=np.uint8)
        labeled_mask[segmentation] = 1
        labeled_mask = measure.label(labeled_mask)
        labeled_mask = clear_border(labeled_mask)

        # Loop over each connected component and extract its region props
        for region in measure.regionprops(labeled_mask):
            # Get the bounding box coordinates
            minr, minc, maxr, maxc = region.bbox

            # Calculate the diameter
            diameter = region.major_axis_length * pixel_to_micron
            circularity = (4 * region.area * np.pi) / (region.perimeter ** 2)

            # Filter based on the diameter and circularity thresholds
            if diameter > diameter_threshold and circularity > circularity_threshold:
                # Plot the segmentation with the bounding box
                ax.imshow(image, cmap='gray')
                rect = Rectangle((minc, minr), maxc - minc, maxr - minr,
                                 fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)

                # Add the diameter as text
                # ax.text(minc, minr, f'D: {diameter:.2f}', color='red',
                # verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})

    ax.axis('off')
    if display == True:
        plt.show()

    else:
        plt.savefig('diameter_plot.png', bbox_inches='tight', pad_inches=0)


def ind_mask(masks, diameter_threshold, circularity_threshold, pixel_to_micron, image, display=False):
    filtered_masks = []

    for mask in masks:
        # Extract the segmentation from the mask
        segmentation = mask['segmentation']

        # Create a labeled mask using the segmentation and the 'true' value
        labeled_mask = np.zeros_like(segmentation, dtype=np.uint8)
        labeled_mask[segmentation] = 1
        labeled_mask = measure.label(labeled_mask)

        # Remove segments touching the border
        labeled_mask = clear_border(labeled_mask)

        # Loop over each connected component and extract its region props
        for region in measure.regionprops(labeled_mask):
            # Calculate the diameter
            diameter = region.major_axis_length * pixel_to_micron

            # Calculate the circularity
            circularity = (4 * region.area * np.pi) / (region.perimeter ** 2)

            # Filter based on the diameter and circularity thresholds
            if diameter > diameter_threshold and circularity > circularity_threshold:
                filtered_masks.append((mask, region))
                break  # Found a satisfying mask, no need to process further masks in the loop

    num_plots = len(filtered_masks)
    num_cols = min(3, num_plots)  # Set a maximum of 3 columns
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, (mask, region) in enumerate(filtered_masks):
        # Extract the segmentation from the mask
        segmentation = mask['segmentation']

        # Create a labeled mask using the segmentation and the 'true' value
        labeled_mask = np.zeros_like(segmentation, dtype=np.uint8)
        labeled_mask[segmentation] = 1
        labeled_mask = measure.label(labeled_mask)

        # Remove segments touching the border
        labeled_mask = clear_border(labeled_mask)

        # Create a subplot for the current mask
        ax = axes[i]

        # Show the image
        ax.axis('off')

        ax.imshow(image)

        # Loop over each connected component and extract its region props
        for region in measure.regionprops(labeled_mask):
            # Calculate the diameter
            diameter = region.major_axis_length * pixel_to_micron

            # Calculate the circularity
            circularity = (4 * region.area * np.pi) / (region.perimeter ** 2)

            # Draw a bounding box around the region
            min_row, min_col, max_row, max_col = region.bbox
            rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                 fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

            # Create the caption text
            caption = f"Area: {region.area * pixel_to_micron * pixel_to_micron:.2f}\nPerimeter: {region.perimeter * pixel_to_micron:.2f}\nDiameter: {diameter:.2f}\nCircularity: {circularity:.2f}"

            # Add the caption as title to the subplot
            ax.set_title(caption, color='red', fontsize=10)

    # Hide empty subplots
    for i in range(len(filtered_masks), len(axes)):
        axes[i].axis('off')

    # Adjust subplot spacing
    fig.tight_layout()

    # Show the plot
    if display == True:
        plt.show()
    else:
        plt.savefig('individual_annotated_masks.png',
                    bbox_inches='tight', pad_inches=0)


def stat_sum(diameter_threshold, circularity_threshold, csv_directory):
    stat = pd.read_csv(csv_directory)

    filtered_stat = stat[(stat['diameter'] > diameter_threshold) & (
        stat['circularity'] > circularity_threshold)]

    summary = filtered_stat.describe()

    return summary


def plot_boxplots(diameter_threshold, circularity_threshold, csv_directory):
    stat = pd.read_csv(csv_directory)
    stat = stat[(stat['diameter'] > diameter_threshold) &
                (stat['circularity'] > circularity_threshold)]
    data = [stat["area"].dropna(), stat["perimeter"].dropna(), stat["diameter"].dropna(),
            stat["circularity"].dropna()]

    data_names = ["Area", "Perimeter", "Diameter", "Circularity"]

    # Create a figure and axes
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # Loop over the axes and create a box plot for each dataset
    for i, ax in enumerate(axs):
        ax.boxplot(data[i])

        # Set the title and y-axis label for each subplot
        ax.set_title(data_names[i])
        ax.set_ylabel("Value")

        # Remove x-axis labels and ticks
        ax.set_xticks([])
        ax.set_xticklabels([])

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_psd_from_csv(diameter_threshold, circularity_threshold, num_bins, csv_directory):
    stat = pd.read_csv(csv_directory)

    # Apply diameter and circularity thresholds
    filtered_stat = stat[(stat['diameter'] > diameter_threshold) & (
        stat['circularity'] > circularity_threshold)]

    f, ax = plt.subplots()
    ax1 = sns.histplot(data=filtered_stat["diameter"], bins=num_bins, kde=True,
                       line_kws={"linewidth": 2, 'color': '#b44a46'})
    ax1 = plt.twinx()
    sns.ecdfplot(data=filtered_stat, x="diameter", ax=ax1, stat="proportion", color='red',
                 linewidth=2)

    ax.set(xlabel='Particle size', ylabel='Number of particles')
    ax1.set_ylabel('Cumulative frequency of particles')
    plt.xticks(fontsize=12)
    plt.title("Particle size distribution")
    plt.show()


def plot_psd(diameter_threshold, circularity_threshold, num_bins, segments):
    stat = pd.DataFrame(segments)

    # Apply diameter and circularity thresholds
    filtered_stat = stat[(stat['diameter'] > diameter_threshold) & (
        stat['circularity'] > circularity_threshold)]

    f, ax = plt.subplots()
    ax1 = sns.histplot(data=filtered_stat["diameter"], bins=num_bins, kde=True,
                       line_kws={"linewidth": 2, 'color': '#b44a46'})
    ax1 = plt.twinx()
    sns.ecdfplot(data=filtered_stat, x="diameter", ax=ax1, stat="proportion", color='red',
                 linewidth=2)

    ax.set(xlabel='Particle size', ylabel='Number of particles')
    ax1.set_ylabel('Cumulative frequency of particles')
    plt.xticks(fontsize=12)
    plt.title("Particle size distribution")
    plt.show()


def plot_psd_bins(diameter_threshold, circularity_threshold, bins, segments):

    bin_edges, counts, cumulative_area = get_psd_data(
        diameter_threshold, circularity_threshold, bins, segments, reverse_cumulative=True)

    # append 0 for bottom
    plot_bins = [0] + bins[:]

    # Create equal spacing for plotting
    equal_spacing = np.arange(len(plot_bins))

    # Plot the histogram on the primary axis
    f, ax = plt.subplots()
    ax.bar(equal_spacing[:-1], counts, align='center',
           edgecolor='black', color='skyblue')

    # Set the x-axis limits and ticks to match the equal spacing
    ax.set_xlim(min(equal_spacing), max(equal_spacing))
    ax.set_xticks(equal_spacing)
    ax.set_xticklabels([''] * len(equal_spacing))  # Hide the edge tick labels

    # Manually set the positions of the tick labels to be at the midpoints
    for i, (midpoint, label) in enumerate(zip(equal_spacing, [f'{edge:.2f}' for edge in bin_edges[1:]]), start=1):
        ax.text(midpoint, -0.04, str(float(label)/1000), ha='center',
                va='bottom', fontsize=8, transform=ax.get_xaxis_transform())

    # Create the secondary axis for the cumulative area plot
    ax1 = ax.twinx()
    ax1.plot(equal_spacing[::-1][:-1],
             cumulative_area, color='red', linewidth=2)

    # Convert the y-axis of the cumulative area plot to percentage
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))

    # Set the labels for both axes
    # Adjust the labelpad value to move the title lower
    ax.set_xlabel('Particle size (mm)', labelpad=20)
    ax.set_ylabel('% Retained (Area %)')
    ax1.set_ylabel('Cumulative % passing (Area %)')

    plt.title("Particle size distribution")

    # Show the plot
    plt.show()


def plot_psd_bins2(diameter_threshold, circularity_threshold, bins, segments, fileName):
    bin_edges, counts, cumulative_area = get_psd_data(
        diameter_threshold, circularity_threshold, bins, segments, reverse_cumulative=True)

    # append 0 for bottom
    plot_bins = [0] + bins[:]

    # Create equal spacing for plotting
    equal_spacing = np.arange(len(plot_bins))

    # Plot the histogram on the primary axis
    f, ax = plt.subplots(figsize=(12, 8))
    # Add more padding to prevent cutoff
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.9)
    ax.bar(equal_spacing[:-1], counts, align='center',
           edgecolor='black', color='skyblue', label='% Retained (Area %)')

    # Set the x-axis limits and ticks to match the equal spacing
    ax.set_xlim(min(equal_spacing), max(equal_spacing))
    ax.set_xticks(equal_spacing)
    ax.set_xticklabels([''] * len(equal_spacing))  # Hide the edge tick labels

    # Manually set the positions of the tick labels to be at the midpoints
    for i, (midpoint, label) in enumerate(zip(equal_spacing, [f'{edge:.2f}' for edge in bin_edges[1:]]), start=1):
        ax.text(midpoint, -0.04, str(float(label) / 1000), ha='center',
                va='bottom', fontsize=8, transform=ax.get_xaxis_transform())

    # Create the secondary axis for the cumulative area plot
    ax1 = ax.twinx()
    ax1.plot(equal_spacing[::-1][:-1],
             cumulative_area, color='red', linewidth=2, label='Cumulative % passing (Area %)')

    # Convert the y-axis of the cumulative area plot to percentage
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))

    # Set the labels for both axes
    # Adjust the labelpad value to move the title lower
    ax.set_xlabel('Particle size (mm)', labelpad=20)
    ax.set_ylabel('% Retained (Area %)', labelpad=20)
    ax1.set_ylabel('Cumulative % passing (Area %)', labelpad=10)
    # Get handles and labels from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()

    # Create a unified legend below the plot
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='center', bbox_to_anchor=(0.5, -0.25),
               ncol=1, frameon=True, fancybox=True, shadow=True)
    # Adjust the layout to prevent legend cutoff
    plt.tight_layout()
    # Save the plot as an image file with adjusted figure size to accommodate legend
    plt.gcf().set_size_inches(8, 7)  # Adjust figure size if needed
    plt.title("Particle size distribution", pad=20)
    # Adjust the layout to prevent legend cutoff
    # Save the plot to the path constructed
    plt.savefig(fileName, bbox_inches='tight', dpi=300, pad_inches=0.5)

    plt.close()

 # This is for the diamater distributions


def plot_psd_bins4(diameter_threshold, circularity_threshold, bins, segments, fileName):
    # Obtain particle size distribution data
    bin_edges, counts, cumulative_area = custom_psd_data1(
        diameter_threshold, circularity_threshold, bins, segments, reverse_cumulative=True)

    # Reverse cumulative_area and skip the first data point
    # Start from the second element
    cumulative_area = cumulative_area[::-1][1:]

    # Skip the first data point for counts as well
    counts = counts[1:]  # Start from the second element

    # Plot the histogram on the primary axis
    f, ax = plt.subplots(figsize=(12, 8))
    # Add more padding to prevent cutoff
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.9)
    # Distribute bin_edges evenly
    equal_spacing = np.linspace(0, 1, len(counts))
    # Calculate the width of each bin
    bin_width = equal_spacing[1] - equal_spacing[0]

    # Draw the bars with a width of 80% of the equal spacing to ensure gaps
    ax.bar(equal_spacing, counts, width=bin_width * 0.8, align='center',
           edgecolor='black', color='skyblue', label='% Retained (Diameter %)')

    # Set the ticks and labels for the x-axis based on bin boundaries
    ax.set_xticks(equal_spacing)
    # Convert edge to mm
    ax.set_xticklabels([f'{edge/1000}' for edge in bin_edges])

    # Create a secondary y-axis for the cumulative percentage
    ax1 = ax.twinx()
    ax1.plot(equal_spacing, cumulative_area, 'o-', color='red', linewidth=2,
             label='Cumulative % passing (Diameter %)')  # Ensure points are connected by lines

    # Format the secondary y-axis as percentage
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))

    # Set labels for both axes
    ax.set_xlabel('Particle size (mm)', labelpad=20)
    ax.set_ylabel('% Retained (Diameter %)', labelpad=20)
    ax1.set_ylabel('Cumulative % passing (Diameter %)', labelpad=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()

    # Create a unified legend below the plot
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='center', bbox_to_anchor=(0.5, -0.25),
               ncol=1, frameon=True, fancybox=True, shadow=True)
    # Adjust the layout to prevent legend cutoff
    plt.tight_layout()
    # Save the plot as an image file with adjusted figure size to accommodate legend
    plt.gcf().set_size_inches(8, 7)  # Adjust figure size if needed
    plt.title("Particle size distribution", pad=20)

    # Save the plot to the path constructed
    plt.savefig(fileName, bbox_inches='tight', dpi=300, pad_inches=0.5)

    plt.close()


def plot_psd_bins1(diameter_threshold, circularity_threshold, bins, segments):
    stat = pd.DataFrame(segments)

    # Apply diameter and circularity thresholds
    filtered_stat = stat[(stat['diameter'] > diameter_threshold) & (
        stat['circularity'] > circularity_threshold)]

    # Calculate the total area
    total_area = filtered_stat['area'].sum()

    # Calculate the histogram of areas
    counts, bin_edges = np.histogram(
        filtered_stat['diameter'], bins=bins, weights=filtered_stat['area'])
    counts = counts / total_area * 100  # Convert counts to percentages of total area

    # Populate cumulative passing
    cumulative_area = []
    counts_reversed = counts[::-1]
    for i, count in enumerate(counts_reversed):
        if i == 0:
            cumulative_area.append(100 - count)
        else:
            cumulative_area.append(cumulative_area[i - 1] - count)

    # Create equal spacing for plotting
    equal_spacing = np.arange(len(bins))

    # Plot the histogram on the primary axis
    f, ax = plt.subplots()
    ax.bar(equal_spacing[:-1], counts, align='center',
           edgecolor='black', color='skyblue')

    # Set the x-axis limits and ticks to match the equal spacing
    ax.set_xlim(min(equal_spacing), max(equal_spacing))
    ax.set_xticks(equal_spacing)
    ax.set_xticklabels([''] * len(equal_spacing))  # Hide the edge tick labels

    # Manually set the positions of the tick labels to be at the midpoints
    for i, (midpoint, label) in enumerate(zip(equal_spacing, [f'{edge:.2f}' for edge in bin_edges[1:]]), start=1):
        ax.text(midpoint, -0.04, str(float(label)/1000), ha='center',
                va='bottom', fontsize=8, transform=ax.get_xaxis_transform())

    # Create the secondary axis for the cumulative area plot
    ax1 = ax.twinx()
    ax1.plot(equal_spacing[::-1][:-1],
             cumulative_area, color='red', linewidth=2)

    # Convert the y-axis of the cumulative area plot to percentage
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))

    # Set the labels for both axes
    # Adjust the labelpad value to move the title lower
    ax.set_xlabel('Particle size (mm)', labelpad=20)
    ax.set_ylabel('% Retained (Area %)')
    ax1.set_ylabel('Cumulative % passing (Area %)')

    plt.title("Particle size distribution")

    # Show the plot
    plt.show()


def calculate_totalArea(diameter_threshold, circularity_threshold, segments):
    stat = pd.DataFrame(segments)

    # Apply diameter and circularity thresholds
    # Check if both diameter_threshold and circularity_threshold are non-zero
    if diameter_threshold > 0 and circularity_threshold > 0:
        filtered_stat = stat[(stat['diameter'] < diameter_threshold) & (
            stat['circularity'] < circularity_threshold)]
    elif diameter_threshold > 0:
        filtered_stat = stat[stat['diameter'] < diameter_threshold]
    elif circularity_threshold > 0:
        filtered_stat = stat[stat['circularity'] < circularity_threshold]
    else:
        # If both thresholds are 0, no filtering is applied
        filtered_stat = stat

    # Calculate the total area
    total_area = filtered_stat['area'].sum()
    return total_area


def adjustSegments(stat):
    # Identify rows where diameter is 0.0 and area is non-zero
    mask = (stat['diameter'] == 0.0) & (stat['area'] != 0.0)

    # Adjust the diameter for these segments
    stat.loc[mask, 'diameter'] = np.sqrt(stat.loc[mask, 'area'] / np.pi) / 2

    return stat


def get_psd_data(diameter_threshold, circularity_threshold, bins, segments, reverse_cumulative=True):

    # sort the bins first
    # because we are calculating bins we need to add a 0 at the front
    plot_bins = [0] + bins[:]

    stat = pd.DataFrame(segments)
    stat = adjustSegments(stat)

    # Apply diameter and circularity thresholds
    if diameter_threshold > 0 and circularity_threshold > 0:
        filtered_stat = stat[(stat['diameter'] < diameter_threshold) & (
            stat['circularity'] < circularity_threshold)]
    elif diameter_threshold > 0:
        filtered_stat = stat[stat['diameter'] < diameter_threshold]
    elif circularity_threshold > 0:
        filtered_stat = stat[stat['circularity'] < circularity_threshold]
    else:
        # If both thresholds are 0, no filtering is applied
        filtered_stat = stat

    # Calculate the total area
    total_area = filtered_stat['area'].sum()

    # Calculate the histogram of areas
    counts, bin_edges = np.histogram(
        filtered_stat['diameter'], bins=plot_bins, weights=filtered_stat['area'])

    counts = counts / total_area * 100  # Convert counts to percentages of total area

    # Populate cumulative passing
    cumulative_area = []

    counts_reversed = counts[::-1]
    for i, count in enumerate(counts_reversed):
        if i == 0:
            cumulative_area.append(100 - count)
        else:
            cumulative_area.append(cumulative_area[i - 1] - count)

    if reverse_cumulative == True:
        return bin_edges, counts, cumulative_area
    else:
        return bin_edges, counts, cumulative_area


def custom_psd_data1(diameter_threshold, circularity_threshold, bins, segments, reverse_cumulative=True):

    # sort the bins first
    # because we are calculating bins we need to add a 0 at the front
    plot_bins = [0] + bins[:]

    stat = pd.DataFrame(segments)
    stat = adjustSegments(stat)

    # Apply diameter and circularity thresholds
    if diameter_threshold > 0 and circularity_threshold > 0:
        filtered_stat = stat[(stat['diameter'] < diameter_threshold) & (
            stat['circularity'] < circularity_threshold)]
    elif diameter_threshold > 0:
        filtered_stat = stat[stat['diameter'] < diameter_threshold]
    elif circularity_threshold > 0:
        filtered_stat = stat[stat['circularity'] < circularity_threshold]
    else:
        # If both thresholds are 0, no filtering is applied
        filtered_stat = stat

    # Calculate the total particles
    total_particles = len(filtered_stat)
    counts = [0] * len(plot_bins)
    cumulative = [0] * len(plot_bins)

    bin_labels = [f"Diameter < {bins[0]}"]
    for i in range(1, len(bins)):
        bin_labels.append(f"{bins[i - 1]} <= Diameter < {bins[i]}")
    bin_labels.append(f"Diameter >= {bins[-1]}")

    # Initialize bin label dictionary
    counts_dict = {label: 0 for label in bin_labels}

    # Iterate all diameters in filtered_stat
    for diameter in filtered_stat['diameter']:
        if diameter < bins[0]:
            counts[0] += 1
            counts_dict[bin_labels[0]] += 1
        elif diameter >= bins[-1]:  # The biggest bin value and above
            counts[-1] += 1
            counts_dict[bin_labels[-1]] += 1
        else:
            for i in range(1, len(bins)):
                if bins[i - 1] <= diameter < bins[i]:
                    counts[i] += 1
                    counts_dict[bin_labels[i]] += 1
                    break
    count_percent = []
    for count in counts:
        percent = count/total_particles*100
        count_percent.append(percent)

    print("counts:", counts)
    print("counts_dict", counts_dict)
    print("count percent: ", count_percent)

    cumulative_percent = []
    for diameter in filtered_stat['diameter']:
        # cumulative count
        for i in range(len(bins)):
            if diameter < bins[i]:
                cumulative[i+1] += 1
    for cumlative_particle in cumulative:
        percent = cumlative_particle / total_particles * 100
        cumulative_percent.append(percent)
    cumulative_percent.reverse()
    print("cumulative_percent", cumulative_percent)

    if reverse_cumulative == True:
        return bins, count_percent, cumulative_percent
    else:
        return bins, count_percent, cumulative_percent


def plot_psd_bins3(diameter_threshold, circularity_threshold, bins, segments):
    stat = pd.DataFrame(segments)
    stat = adjustSegments(stat)

    # Apply diameter and circularity thresholds
    if diameter_threshold > 0 and circularity_threshold > 0:
        filtered_stat = stat[(stat['diameter'] < diameter_threshold) & (
            stat['circularity'] < circularity_threshold)]
    elif diameter_threshold > 0:
        filtered_stat = stat[stat['diameter'] < diameter_threshold]
    elif circularity_threshold > 0:
        filtered_stat = stat[stat['circularity'] < circularity_threshold]
    else:
        filtered_stat = stat

    # Calculate the total area
    total_area = filtered_stat['area'].sum()

    # Calculate the histogram of areas
    counts, bin_edges = np.histogram(
        filtered_stat['diameter'], bins=bins, weights=filtered_stat['area'])
    counts = counts / total_area * 100  # Convert counts to percentages of total area

    # Calculate the cumulative area (reversed)
    # Cumulative sum in reverse order
    cumulative_area = np.cumsum(counts[::-1])[::-1]
    # Subtract from 100 to get cumulative percentage
    cumulative_area = 100 - cumulative_area

    # Calculate the midpoints of the bins
    midpoints = bin_edges[:-1] + np.diff(bin_edges) / 2

    # Plot the histogram on the primary axis
    f, ax = plt.subplots()
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
           align='edge', edgecolor='black', color='skyblue')

    # Set the x-axis limits and ticks to match the bin edges
    ax.set_xlim(min(bins), max(bins))
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([''] * len(bin_edges))  # Hide the edge tick labels

    # Manually set the positions of the tick labels to be at the midpoints
    for i, (midpoint, label) in enumerate(zip(midpoints, [f'{edge:.2f}' for edge in bin_edges[1:]]), start=1):
        ax.text(midpoint, -0.04, label, ha='center', va='bottom',
                fontsize=8, transform=ax.get_xaxis_transform())

    # Create the secondary axis for the cumulative area plot
    ax1 = ax.twinx()
    ax1.plot(midpoints[::-1], cumulative_area, color='red', linewidth=2)

    # Convert the y-axis of the cumulative area plot to percentage
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))

    # Set the labels for both axes
    ax.set_xlabel('Particle size', labelpad=20)
    ax.set_ylabel('Percentage of total particle area')
    ax1.set_ylabel('Cumulative percentage of total particle area')

    plt.title("Particle size distribution")

    # Show the plot
    plt.show()


# def save_psd(diameter_threshold, circularity_threshold, bins, segments, filename):
#     stat = pd.DataFrame(segments)

#     # Apply diameter and circularity thresholds
#     filtered_stat = stat[(stat['diameter'] > diameter_threshold) & (stat['circularity'] > circularity_threshold)]

#     # Sort the bins in order
#     bins.sort()


def save_psd_as_txt(id, bins, cumulative, differential, csvpath):

    # prepare for export arrangement
    # reverse order for bins
    bins_export = [0] + bins[:]
    bins_export[0] = "Bottom"
    bins_export.reverse()
    differential.reverse()
    cumulative.reverse()

    # csvpath = os.path.join(csv_directory, f"{id}_distribution.txt")

    # write to csv file id, bins, cumulative, differential
    with open(csvpath, 'w', newline='') as csvfile:
        data = [id] + bins_export + ['% Passing'] + \
            cumulative + ['% Retained'] + differential
        writer = csv.writer(csvfile)
        writer.writerow(data)
    return


def save_psd_as_txt_normal(id, bins, cumulative, differential, csv_directory):

    # prepare for export arrangement
    # reverse order for bins
    bins_export = [0] + bins[:]
    bins_export[0] = "Bottom"
    bins_export.reverse()
    differential.reverse()
    cumulative.reverse()

    csvpath = os.path.join(csv_directory, f"{id}_normalBin_distribution.txt")

    # write to csv file id, bins, cumulative, differential
    with open(csvpath, 'w', newline='') as csvfile:
        data = [id] + bins_export + ['% Passing'] + \
            cumulative + ['% Retained'] + differential
        writer = csv.writer(csvfile)
        writer.writerow(data)
    return


def plot_cir(diameter_threshold, circularity_threshold, num_bins, csv_directory):
    stat = pd.read_csv(csv_directory)

    # Apply diameter and circularity thresholds
    filtered_stat = stat[(stat['diameter'] > diameter_threshold) & (
        stat['circularity'] > circularity_threshold)]

    f, ax = plt.subplots()
    ax1 = sns.histplot(data=filtered_stat["circularity"], bins=num_bins, kde=True,
                       line_kws={"linewidth": 2, 'color': '#b44a46'})

    ax.set(xlabel='Circularity', ylabel='Number of particles')
    plt.xticks(fontsize=12)
    plt.title("Circularity distribution")
    plt.show()

# Characterisation section


def line_scan(image, image_bse, masks, circularity_threshold, min_area, csv_file, pixel_to_micron, line_distance_man, plot=False):
    # Resize the BSE image to match the input image
    image_bse = resize(image_bse, image.shape)

    mask_details = []

    # Open the CSV file for writing mask details
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['Mask', 'Area', 'Circularity', 'Type', 'BBox', 'Predicted IOU', 'Point Coords',
                      'Stability Score', 'Perimeter', 'Diameter']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the CSV header
        writer.writeheader()

        cenosphere_images = []
        solidsphere_images = []

        # Process each mask
        for i, mask in enumerate(masks):
            segmentation_array = mask['segmentation']
            area = mask['area']
            predicted_iou = mask['predicted_iou']
            bbox = mask['bbox']  # Bounding box [x, y, w, h]
            point_coords = mask['point_coords']
            stability_score = mask['stability_score']
            crop_box = mask['crop_box']

            # Crop the segmented image using the bounding box
            x, y, w, h = bbox
            segmented_image = cv2.bitwise_and(
                image_bse, image_bse, mask=segmentation_array.astype(np.uint8))
            cropped_segmented_image = segmented_image[y:y+h, x:x+w]

            # Function to calculate circularity of a mask
            def calculate_circularity(segmentation_array):
                contours, _ = cv2.findContours(segmentation_array.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                perimeter = cv2.arcLength(contours[0], True)
                area = cv2.contourArea(contours[0])
                if perimeter == 0:
                    return 0
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                return circularity

            # Function for the polynomial fit
            def polynomial_func(x, a, b, c, d, e):
                return a * x**4 + b * x**3 + c * x**2 + d * x + e

            # Calculate circularity
            circularity = calculate_circularity(segmentation_array)

            # Check circularity and area thresholds
            if circularity > circularity_threshold and area > min_area:
                height = cropped_segmented_image.shape[0]
                # Start line at 10% of the height
                start_line = int(0.1 * height)
                end_line = int(0.9 * height)  # End line at 90% of the height
                line_distance = end_line - start_line

                num_line_scans = int(
                    line_distance / (line_distance_man / pixel_to_micron))
                line_scan_indices = np.linspace(
                    start_line, end_line, num_line_scans, dtype=int)
                line_scans = []  # Store line scan data for each mask

                # Perform line scanning
                for line_index in line_scan_indices:
                    line_pixel_values = cropped_segmented_image[line_index, :].flatten(
                    )
                    x = np.arange(len(line_pixel_values))

                    # Fit a polynomial curve to the line scan data
                    popt_line, pcov_line = curve_fit(
                        polynomial_func, x, line_pixel_values)
                    fitted_curve_line = polynomial_func(x, *popt_line)

                    # Find the maxima or minima points of the fitted curve
                    line_extremum_index = np.argmax(
                        fitted_curve_line) if popt_line[0] < 0 else np.argmin(fitted_curve_line)
                    line_extremum_x = x[line_extremum_index]
                    line_extremum_y = fitted_curve_line[line_extremum_index]

                    line_maxima_indices = argrelextrema(
                        fitted_curve_line, np.greater)[0]
                    line_minima_indices = argrelextrema(
                        fitted_curve_line, np.less)[0]

                    # Store line scan data
                    line_scans.append({
                        'line_pixel_values': line_pixel_values,
                        'fitted_curve_line': fitted_curve_line,
                        'line_extremum_x': line_extremum_x,
                        'line_extremum_y': line_extremum_y,
                        'line_maxima_indices': line_maxima_indices,
                        'line_minima_indices': line_minima_indices
                    })

                # Count the total number of line_minima_indices
                total_line_minima_indices = sum(
                    len(scan['line_minima_indices']) for scan in line_scans)

                # Classify the segment
                segment_type = 'cenosphere' if total_line_minima_indices > 0 else 'solid sphere'

                # Save additional region properties
                labeled_mask = np.zeros_like(
                    segmentation_array, dtype=np.uint8)
                labeled_mask[segmentation_array] = 1
                labeled_mask = measure.label(labeled_mask)
                cleared_mask = clear_border(labeled_mask)
                region_props = measure.regionprops(cleared_mask)

                for region in region_props:
                    # Convert area, perimeter, and diameter from pixels to micrometers
                    area_pixels = region.area
                    perimeter_pixels = region.perimeter
                    diameter_pixels = region.major_axis_length

                    # Convert measurements to microns
                    area_2 = area_pixels * pixel_to_micron**2
                    perimeter = perimeter_pixels * pixel_to_micron
                    diameter = diameter_pixels * pixel_to_micron

                    # Append mask details to the list
                    mask_details.append({
                        'Mask': i + 1,
                        'Area': area * pixel_to_micron**2,
                        'Circularity': circularity,
                        'BBox': bbox,
                        'Predicted IOU': predicted_iou,
                        'Point Coords': point_coords,
                        'Stability Score': stability_score,
                        'Perimeter': perimeter,
                        'Diameter': diameter,
                        'Type': segment_type
                    })

                    # Write mask details to CSV file
                    writer.writerow({
                        'Mask': i + 1, 'Area': area, 'Circularity': circularity, 'BBox': bbox,
                        'Predicted IOU': predicted_iou, 'Point Coords': point_coords,
                        'Stability Score': stability_score, 'Perimeter': perimeter,
                        'Diameter': diameter, 'Type': segment_type
                    })


def plot_segment_bounding_boxes(csv_file, segment_types, image):
    # Read the CSV file and extract the relevant data
    mask_details = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mask_details.append(row)

    # Filter the mask details based on the segment types
    filtered_mask_details = [
        mask for mask in mask_details if mask['Type'] in segment_types]

    # Create a single plot
    fig, ax = plt.subplots()

    # Load and plot the corresponding images with bounding boxes
    colors = ['r', 'b']  # Colors for each segment type

    for idx, segment_type in enumerate(segment_types):
        segments = [
            mask for mask in filtered_mask_details if mask['Type'] == segment_type]

        for mask in segments:
            bbox = eval(mask['BBox'])

            # Plot the image
            plt.imshow(image, cmap='gray')

            # Create a rectangle patch for the bounding box
            x, y, w, h = bbox
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor=colors[idx], facecolor='none')

            # Add the rectangle patch to the plot
            ax.add_patch(rect)

    # Set the title
    # plt.title(f"Segment Types: {', '.join(segment_types)}")

    # Create custom legend patches
    legend_patches = [patches.Patch(
        facecolor='none', edgecolor=colors[i], label=segment_types[i]) for i in range(len(segment_types))]

    # Add the legend with the custom patches
    plt.legend(handles=legend_patches, bbox_to_anchor=(0.5, -0.02),
               loc='upper center', ncol=len(segment_types))

    plt.axis('off')
    plt.tight_layout()

    # Show the plot
    plt.show()


def psd_spheres(csv_file):
    cenosphere_sizes = []
    solid_sizes = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mask_type = row['Type']
            diameter = float(row['Diameter'])

            if mask_type == 'cenosphere':
                cenosphere_sizes.append(diameter)
            else:
                solid_sizes.append(diameter)

    fig, ax = plt.subplots()
    sns.histplot(cenosphere_sizes, bins=10, kde=True,
                 color='#FFBE86', label='Cenospheres', ax=ax)
    sns.histplot(solid_sizes, bins=10, kde=True,
                 color='#8EBAD9', label='Solid Spheres', ax=ax)
    ax.set_xlabel('Diameter (µm)')
    ax.set_ylabel('Count')
    ax.legend()
    plt.show()


def box_plots_spheres(csv_file):
    cenosphere_sizes = []
    solid_sizes = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mask_type = row['Type']
            diameter = float(row['Diameter'])

            if mask_type == 'cenosphere':
                cenosphere_sizes.append(diameter)
            else:
                solid_sizes.append(diameter)

    fig, ax = plt.subplots()
    sns.boxplot(data=[cenosphere_sizes, solid_sizes], ax=ax)
    ax.set_xticklabels(['Cenospheres', 'Solid Spheres'])
    ax.set_ylabel('Diameter (µm)')
    ax.set_title('Box Plot - Cenospheres vs Solid Spheres')
    plt.show()


def save_segments_as_csv(txt_filename, csv_filename, diameter_threshold):
    try:
        filtered_segments = []

        # Load the JSON data from the text file
        with open(txt_filename, 'r') as file:
            content = file.read()

        data = json.loads(content)

        # Validate that the data is a list of dictionaries
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            print("Error: The data format is incorrect.")
            return
        zero_diameter_segments = [
            segment for segment in data if segment.get("diameter", 0) == 0
        ]

        for segment in zero_diameter_segments:
            if segment.get("diameter", 0) == 0 and segment.get("area", 0) != 0:
                # Calculate the diameter from the area using the formula
                calculated_diameter = np.sqrt(segment['area'] / np.pi) * 2
                segment['diameter'] = calculated_diameter

        # Filter segments based on the diameter threshold
        filtered_segments = [segment for segment in data if segment.get(
            "diameter", float('inf')) < diameter_threshold]

        # Get fieldnames for CSV from the first segment
        if filtered_segments:
            fieldnames = filtered_segments[0].keys()

            # Write the filtered segments to the CSV file
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in filtered_segments:
                    writer.writerow(row)

            print(
                f"Saved {len(filtered_segments)} segments to {csv_filename} with diameter < {diameter_threshold}")
        else:
            print("No segments found with diameter below the threshold.")

        return filtered_segments

    except Exception as e:
        print(f"Error while processing the file: {e}")
