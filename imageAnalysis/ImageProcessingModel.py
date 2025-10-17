from logger_config import get_logger
import traceback
import math
import time
import io
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import sys

sys.path.append(os.path.join(os.getcwd(), "imagePreprocessing"))
ROI = __import__("ROISelector")
logger = get_logger("ImageProcess")


class ImageProcessingModel:
    def __init__(self, image_folder_path, sampleID):
        """
        Initializes the ImageProcessingModel with the provided folder path and sample ID.
        It searches for an image file with the given sample ID and supported extensions (.png, .bmp).

        Inputs:
            image_folder_path (str): Path to the folder containing images.
            sampleID (str): The sample ID to identify the image, with no extension 

        Outputs:None
        """
        self.sampleID = sampleID
        self.image_folder_path = image_folder_path
        self.file_extensions = ['.png', '.bmp', '.jpg']
        self.imagePath = None
        self.image_extension = None
        self.raw_imagePath = None  # Attribute to store the path of the raw image copy
        self.evenLightingImagePath = None
        self.colorCorrection_imagePath = None
        # Loop through extensions and check for existence
        for ext in self.file_extensions:
            self.imageName = f"{self.sampleID}{ext}"
            self.imagePath = os.path.join(image_folder_path, self.imageName)

            if os.path.exists(self.imagePath):
                self.image_extension = ext
                logger.trace(f"Image found: {self.imagePath}")
                break
        else:
            # If no file with the listed extensions is found, raise an error
            raise FileNotFoundError(
                f"No file with extensions {self.file_extensions} found for {self.sampleID} in folder {image_folder_path}")

    def getImagePath(self):
        """
        Returns the full path of the image file.

        Inputs:None

        Outputs:str: The full path to the image file.
        """
        return self.imagePath

    def getImageFolder(self):
        """
        Returns the path of the folder containing the image.

        Inputs:None

        Outputs:str: The folder path where the image is located.
        """
        return self.image_folder_path

    def showImage(self):
        """
        Displays the image using matplotlib. Converts the image to RGB and shows it.

        Inputs:None

        Outputs: Show the image

        """
        if os.path.exists(self.imagePath):
            image = Image.open(self.imagePath)
            image = np.array(image.convert("RGB"))
            plt.imshow(image)  # Display the image
            plt.axis('off')  # Optional: Turn off the axis for a cleaner view
            plt.show()  # Show the image
        else:
            logger.trace(
                f"Error: Image {self.imageName} not found at {self.imagePath}")

    def getWidth(self):
        """
        Returns width of an image in the folder with the same sample ID.
        Useful for determining the mm/pixel ratio.

        Inputs:None

        Outputs:
            int:  width of the images found.
        """
        try:
            with Image.open(self.imagePath) as img:
                width, _ = img.size
            return width
        except Exception as e:
            logger.trace(f"Error opening image at {self.imagePath}: {e}")
            return None

    def getIntensity(self):
        """
        Calculates the average intensity (grayscale) of the image.

        Inputs:None

        Outputs: float: The average intensity value of the image.
        """
        image = Image.open(self.imagePath)
        image_np = np.array(image)
        # Convert RGB to BGR for OpenCV compatibility
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_image)

    def overlayImage(self):
        """
        Overlays the image on itself 10 times to improve the contrast of the image,
        especially the borders of rocks, and resizes it if the size exceeds 8MB.

        Inputs:None

        Outputs:None
        """
        try:
            if not os.path.exists(self.imagePath):
                logger.trace(
                    f"Error: Image {self.imageName} not found at {self.imagePath}")
                return

            image_size_mb = os.path.getsize(
                self.imagePath) / (1024 * 1024)  # Size in MB

            base_image = Image.open(self.imagePath).convert("RGBA")

            # Resize image if size is greater than 8MB
            if image_size_mb > 8:
                # Square root to maintain aspect ratio
                scale_factor = (8 / image_size_mb) ** 0.5

                width, height = base_image.size
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                base_image = base_image.resize(
                    (new_width, new_height), Image.LANCZOS)
                logger.trace(
                    f"Image size was over 8MB, resized to {new_width}x{new_height}.")

                image_size_mb = base_image.tell() / (1024 * 1024)

                # If still too large, reduce it further
                while image_size_mb > 8:
                    width, height = base_image.size
                    base_image = base_image.resize(
                        (width // 2, height // 2), Image.LANCZOS)
                    image_size_mb = base_image.tell() / (1024 * 1024)
                    logger.trace(
                        f"Still too large, further resized to {width // 2}x{height // 2}. Current size: {image_size_mb:.2f}MB")

            final_image = base_image.copy()

            # Overlay the image on itself 10 times
            for _ in range(10):
                final_image = Image.alpha_composite(final_image, base_image)

            # Save the base and final overlaid images
            if not self.imagePath.lower().endswith('.png'):
                base_image_path = os.path.join(
                    self.image_folder_path, f"base_image_{self.sampleID}.png")
                self.safe_save_image(base_image, base_image_path)
                #base_image.save(base_image_path)
                logger.trace(f"Base image saved as: {base_image_path}")

            # Save the final overlaid image with a new name
            final_image_name = "final_" + self.sampleID + ".png"
            self.imagePath = os.path.join(
                self.image_folder_path, final_image_name)
            self.safe_save_image(final_image, self.imagePath)
            #final_image.save(self.imagePath)
            logger.trace(f"Final overlaid image saved: {self.sampleID}")
            logger.trace(f"Final overlaid image saved as: {self.imagePath}")
        except Exception as e:
            logger.error(
                f"Error occurred in over lay  : {str(e)},sample_id: {self.sampleID}")
            logger.error(
                f"Error for over lay :{self.sampleID} Traceback: {traceback.format_exc()}")
            raise

    def safe_save_image(self, image, save_path):
        """添加这个方法到您的类中"""
        import io

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 尝试直接保存
            image.save(save_path, "PNG")

        except AttributeError as e:
            if "fileno" in str(e):
                # 使用BytesIO绕过fileno问题
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')

                with open(save_path, 'wb') as f:
                    f.write(buffer.getvalue())
            else:
                raise e
    def pureOverlayImage(self, baseImage, flag):
        """
        Overlays the image on itself 10 times to improve the contrast of the image
        and saves the resulting image into a 'meshingImage' directory under the
        same directory as self.imagePath.

        Inputs:
            baseImage: The base image to process.
            flag: A numeric value used to name the saved image.

        Outputs: None
        """
        # Get the directory from self.imagePath
        image_directory = os.path.dirname(self.imagePath)

        # Create a 'meshingImage' directory if it doesn't exist
        meshing_dir = os.path.join(image_directory, "meshingImage")
        os.makedirs(meshing_dir, exist_ok=True)

        # Create a copy of the base image for overlaying
        final_image = baseImage.copy()

        # Overlay the image on itself 10 times
        for _ in range(10):
            final_image = Image.alpha_composite(final_image, baseImage)

        # Generate the filename based on the flag
        final_image_name = f"meshing_image_{flag}.png"
        final_image_path = os.path.join(meshing_dir, final_image_name)

        # Save the final overlaid image to the meshingImage directory
        final_image.save(final_image_path)
        logger.trace(f"Final overlaid image saved as: {final_image_path}")

    def processImageWithMeshing(self):
        """
        Divides the image into 16 equally sized blocks (4x4 grid)
        and ensures each block is under 8MB.
        Passes each block to the `pureOverlayImage` method for further processing.
        Inputs: None
        Outputs: None
        """

        if not os.path.exists(self.evenLightingImagePath):
            logger.trace(
                f"Error: Image {self.evenLightingImagePath} not found at {self.evenLightingImagePath}")
            return

        # Load the image
        base_image = Image.open(self.evenLightingImagePath).convert("RGBA")
        width, height = base_image.size

        # Define grid size (4x4 -> 16 blocks)
        num_rows = 4
        num_cols = 4
        block_width = width // num_cols
        block_height = height // num_rows

        # List to store blocks
        blocks = []

        # Divide the image into blocks
        for row in range(num_rows):
            for col in range(num_cols):
                left = col * block_width
                right = (col + 1) * block_width
                top = row * block_height
                bottom = (row + 1) * block_height

                # Crop the block
                block = base_image.crop((left, top, right, bottom))

                # Check block size using io.BytesIO
                with io.BytesIO() as temp_buffer:
                    block.save(temp_buffer, format="PNG")
                    block_size_mb = temp_buffer.tell() / (1024 * 1024)

                # Resize the block if it exceeds 8MB
                while block_size_mb > 8:
                    # Calculate scaling factor
                    scale_factor = (8 / block_size_mb) ** 0.5
                    new_width = int(block.width * scale_factor)
                    new_height = int(block.height * scale_factor)
                    block = block.resize(
                        (new_width, new_height), Image.LANCZOS)

                    # Recalculate block size after resizing
                    with io.BytesIO() as temp_buffer:
                        block.save(temp_buffer, format="PNG")
                        block_size_mb = temp_buffer.tell() / (1024 * 1024)

                # Append the resized block to the list
                blocks.append(block)

        logger.trace(
            f"Image divided into {len(blocks)} blocks, and each block is under 8MB.")

        # Process each block with `pureOverlayImage`
        for i in range(1, len(blocks)+1):
            logger.trace(type(blocks[i-1]))
            self.pureOverlayImage(blocks[i-1], i)

    def even_out_lighting(self):
        """
        Even out the lighting in the image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to improve the contrast and smooth out lighting inconsistencies.

        Inputs:None

        Outputs:None
        """
        # Load the image
        try:
            image = cv2.imread(self.imagePath, cv2.IMREAD_COLOR)
            # Convert to LAB color space to separate intensity from color information
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # Split the LAB image into its channels
            l, a, b = cv2.split(lab_image)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

            # Perform a light normalization to smooth out lighting inconsistencies without over-smoothing
            enhanced_image = cv2.normalize(
                enhanced_image, None, 0, 255, cv2.NORM_MINMAX)

            # Apply a slight Gaussian blur to avoid too much noise while keeping details
            final_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

            self.imagePath = os.path.join(
                self.image_folder_path, f"even_lighting_{self.imageName}")
            # self.evenLightingImagePath
            self.evenLightingImagePath = self.imagePath
            cv2.imwrite(self.imagePath, final_image)
            logger.info(f"Evened out lighting picture saved : {self.sampleID}")
            logger.trace(f"Evened out lighting picture saved as : {self.imagePath}")
        except Exception as e:
            logger.error(
                f"Error occurred in even lighting   : {str(e)},sample_id: {self.sampleID}")
            logger.error(
                f"Error for even lighting  :{self.sampleID} Traceback: {traceback.format_exc()}")
            raise

    def even_out_lighting_validation(self, parameter_folder_path):
        """
        Even out the lighting in the image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to improve the contrast and smooth out lighting inconsistencies.

        Inputs:None

        Outputs:None
        """
        # Load the image
        image = cv2.imread(self.imagePath, cv2.IMREAD_COLOR)
        lab_image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2LAB)  # Convert to LAB color space to separate intensity from color information

        l, a, b = cv2.split(lab_image)  # Split the LAB image into its channels

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(
            16, 16))  # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        # Perform a light normalization to smooth out lighting inconsistencies without over-smoothing
        enhanced_image = cv2.normalize(
            enhanced_image, None, 0, 255, cv2.NORM_MINMAX)

        # Apply a slight Gaussian blur to avoid too much noise while keeping details
        final_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
        original_folder_path = self.image_folder_path
        self.image_folder_path = os.path.join(
            original_folder_path, parameter_folder_path)
        os.makedirs(self.image_folder_path, exist_ok=True)
        self.imagePath = os.path.join(
            self.image_folder_path, f"even_lighting_{self.imageName}")
        # self.evenLightingImagePath
        self.evenLightingImagePath = self.imagePath
        cv2.imwrite(self.imagePath, final_image)
        logger.trace(f"Evened out lighting picture saved as : {self.imagePath}")

    # def cropImage(self):
    #     """
    #     Allows the user to manually select a region of interest (ROI) and crop the image to that region.
    #
    #     Inputs:None
    #
    #     Outputs:None
    #     """
    #     roi_selector = ROI.ROISelector(self.imagePath)
    #     cropped_image = roi_selector.select_and_move_roi()
    #     self.imagePath = os.path.join(self.image_folder_path, f"cropped_image_{self.imageName}")
    #     # Save the cropped image to the image folder
    #     cv2.imwrite(self.imagePath, cropped_image)
    def cropImage(self, width=None, height=None, left=None, top=None, defaultHeight=None, defaultWidth=None):
        """
        Crop the image based on specified parameters.

        Inputs:
        - width: Width of the crop rectangle.
        - height: Height of the crop rectangle.
        - left: X coordinate of the top-left corner of the crop rectangle.
        - top: Y coordinate of the top-left corner of the crop rectangle.
        - defaultHeight: Minimum acceptable height of the cropped image.
        - defaultWidth: Minimum acceptable width of the cropped image.

        Outputs: None, updates self.croppedImagePath
        """
        try:
            # Check if all parameters are zero
            if width == 0 and height == 0 and left == 0 and top == 0:
                logger.trace("No cropping needed, all parameters are zero.")
                return

            # Check if only one of width or height is provided
            if (width == 0 and height != 0) or (width != 0 and height == 0):
                logger.trace("Incomplete size parameters, cropping cannot be performed.")
                return

            # Load the image
            image = cv2.imread(self.imagePath)
            if image is None:
                raise ValueError("The image cannot be loaded, check the path.")

            # Ensure provided dimensions are within the image's size and meet default size requirements
            if (left + width > image.shape[1] or top + height > image.shape[0] or
                    (defaultWidth and width < defaultWidth) or
                    (defaultHeight and height < defaultHeight)):
                logger.trace(
                    "Requested dimensions exceed the original image size or do not meet minimum size requirements.")
                return

            # Crop the image
            cropped_image = image[top:top + height, left:left + width]
            cropped_image_name = "crop_" + self.sampleID + ".png"
            # Generate the path to save the cropped image
            self.imagePath = os.path.join(
                self.image_folder_path, cropped_image_name)

            # Save the cropped image
            cv2.imwrite(self.imagePath, cropped_image)
            logger.info(f"Cropped image saved : {self.sampleID}")
            logger.trace(f"Cropped image saved as: {self.imagePath}")
        except Exception as e:
            logger.error(
                f"Error in image cropping : {str(e)},sample_id: {self.sampleID}")
            logger.error(
                f"Error for image crop :{self.sampleID} Traceback: {traceback.format_exc()}")
            raise
    def crop_image_updated(self, corner_points):
        """
        Crops an image using the exact corner points provided.
        Returns a rectangular image containing only the quadrilateral content.

        Args:
            corner_points: List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                           in order: TopLeft, TopRight, BottomRight, BottomLeft
        """
        if len(corner_points) != 4:
            raise ValueError("Four corner points are required.")

        # Load the image
        image = cv2.imread(self.imagePath)
        if image is None:
            raise ValueError("The image cannot be loaded, check the path.")
        
        # Convert OpenCV image to PIL Image
        source_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Find the min/max X and Y coordinates to determine output size
        all_x = [p[0] for p in corner_points]
        all_y = [p[1] for p in corner_points]
        min_x, max_x = int(min(all_x)), int(max(all_x))
        min_y, max_y = int(min(all_y)), int(max(all_y))

        output_width = max_x - min_x
        output_height = max_y - min_y

        if output_width <= 0 or output_height <= 0:
            raise ValueError(f"Invalid crop dimensions: {output_width}x{output_height}")

        # Create the rectangular region from the source image (bounding box)
        image_bbox_crop = source_image.crop((min_x, min_y, max_x, max_y))

        # Create a new transparent image for the output
        output_img = Image.new('RGBA', (output_width, output_height), (0, 0, 0, 0))

        # Create a mask for the quadrilateral
        mask = Image.new('L', (output_width, output_height), 0)  # 'L' for 8-bit grayscale

        # Translate corner points to be relative to the bounding box origin
        translated_corners = [(int(p[0] - min_x), int(p[1] - min_y)) for p in corner_points]

        # Draw the polygon on the mask
        # ImageDraw.Draw(mask).polygon(translated_corners, outline=1, fill=1)
        ImageDraw.Draw(mask).polygon(translated_corners, outline=255, fill=255)
        # Paste the cropped bounding box content onto the output image, using the mask
        output_img.paste(image_bbox_crop, (0, 0), mask)

        # Convert PIL Image to OpenCV format
        cropped_image = cv2.cvtColor(np.array(output_img), cv2.COLOR_RGBA2BGR)

        # Save the cropped image
        cropped_image_name = "crop_" + self.sampleID + ".png"
        self.imagePath = os.path.join(self.image_folder_path, cropped_image_name)
        cv2.imwrite(self.imagePath, cropped_image)

        return cropped_image
    def __kelvin_to_rgb(self, temp_kelvin):

        if temp_kelvin < 1000:
            temp_kelvin = 1000
        elif temp_kelvin > 40000:
            temp_kelvin = 40000

        tmp_internal = temp_kelvin / 100.0

        # red
        if tmp_internal <= 66:
            red = 255
        else:
            tmp_red = 329.698727446 * \
                math.pow(tmp_internal - 60, -0.1332047592)
            if tmp_red < 0:
                red = 0
            elif tmp_red > 255:
                red = 255
            else:
                red = tmp_red

        # green
        if tmp_internal <= 66:
            tmp_green = 99.4708025861 * math.log(tmp_internal) - 161.1195681661
            if tmp_green < 0:
                green = 0
            elif tmp_green > 255:
                green = 255
            else:
                green = tmp_green
        else:
            tmp_green = 288.1221695283 * \
                math.pow(tmp_internal - 60, -0.0755148492)
            if tmp_green < 0:
                green = 0
            elif tmp_green > 255:
                green = 255
            else:
                green = tmp_green

        # blue
        if tmp_internal >= 66:
            blue = 255
        elif tmp_internal <= 19:
            blue = 0
        else:
            tmp_blue = 138.5177312231 * \
                math.log(tmp_internal - 10) - 305.0447927307
            if tmp_blue < 0:
                blue = 0
            elif tmp_blue > 255:
                blue = 255
            else:
                blue = tmp_blue

       # Create an RGB gain matrix.
        return np.clip([red, green, blue], 0, 255)

    def __adjust_temperature(self, image, from_temp, to_temp):
        # Calculate the RGB gains for the original and target color temperatures
        from_rgb = self.__kelvin_to_rgb(from_temp)
        to_rgb = self.__kelvin_to_rgb(to_temp)
        balance = to_rgb / from_rgb
        # Apply the gain
        adjusted = (image * balance).clip(0, 255).astype(np.uint8)

        return adjusted

    def __color_correction(self, imageTemp, adjustedColorTemp):
        image = cv2.imread(self.imagePath)

        result_image = self.__adjust_temperature(
            image, imageTemp, adjustedColorTemp)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # save the result image
        self.colorCorrection_imagePath = os.path.join(os.path.dirname(self.imagePath),
                                                      f"{self.sampleID}_color_correction.png")

        self.imagePath = self.colorCorrection_imagePath
        # Save the result image
        if not cv2.imwrite(self.colorCorrection_imagePath, result_image):
            logger.error(
                f"Failed to save the color corrected image:{self.sampleID}. Check file path and permissions.")
        else:
            logger.info(
                f"Color corrected image saved: sample Id :{self.sampleID}")

    # for image color correction--hardcoded for now
    def colorCorrection(self,temperature, ori_temperature):

        try:
            logger.info(
                f"The original color temperature of the sample ID: {self.sampleID} is {ori_temperature} and the new color lv is {temperature}")
            self.__color_correction(ori_temperature, temperature)
        except Exception as e:
            logger.error(
                f"Error occurred in color correction  : {str(e)},sample_id: {self.sampleID}")
            raise
