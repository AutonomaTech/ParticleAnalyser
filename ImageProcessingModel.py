import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import ROISelector as ROI
import io
import time
import math
import logger_config

logger = logger_config.get_logger(__name__)

class ImageProcessingModel:
    def __init__(self, image_folder_path, sampleID):
        """
        Initializes the ImageProcessingModel with the provided folder path and sample ID.
        It searches for an image file with the given sample ID and supported extensions (.png, .bmp).

        Inputs:
            image_folder_path (str): Path to the folder containing images.
            sampleID (str): The sample ID to identify the image.

        Outputs:None
        """
        self.sampleID = sampleID
        self.image_folder_path = image_folder_path
        self.file_extensions = ['.png', '.bmp']
        self.imagePath = None
        self.image_extension = None
        self.raw_imagePath = None  # Attribute to store the path of the raw image copy
        self.evenLightingImagePath=None
        self.colorCorrection_imagePath = None
        # Loop through extensions and check for existence
        for ext in self.file_extensions:
            self.imageName = f"{self.sampleID}{ext}"
            self.imagePath = os.path.join(image_folder_path, self.imageName)
                
            if os.path.exists(self.imagePath):
                self.image_extension = ext
                print(f"Image found: {self.imagePath}")
                break
        else:
            # If no file with the listed extensions is found, raise an error
            raise FileNotFoundError(f"No file with extensions {self.file_extensions} found for {self.sampleID} in folder {image_folder_path}")

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
            print(f"Error: Image {self.imageName} not found at {self.imagePath}")

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
            print(f"Error opening image at {self.imagePath}: {e}")
            return None
        

    def getIntensity(self):
        """
        Calculates the average intensity (grayscale) of the image.
        
        Inputs:None

        Outputs: float: The average intensity value of the image.
        """
        image = Image.open(self.imagePath)
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV compatibility

        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_image)
        
    def overlayImage(self):
        """
        Overlays the image on itself 10 times to improve the contrast of the image,
        especially the borders of rocks, and resizes it if the size exceeds 8MB.

        Inputs:None

        Outputs:None
        """
        if not os.path.exists(self.imagePath):
            print(f"Error: Image {self.imageName} not found at {self.imagePath}")
            return

        image_size_mb = os.path.getsize(self.imagePath) / (1024 * 1024)  # Size in MB

        base_image = Image.open(self.imagePath).convert("RGBA")

        # Resize image if size is greater than 8MB
        if image_size_mb > 8:
            scale_factor = (8 / image_size_mb) ** 0.5  # Square root to maintain aspect ratio

            width, height = base_image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            base_image = base_image.resize((new_width, new_height), Image.LANCZOS)
            print(f"Image size was over 8MB, resized to {new_width}x{new_height}.")
            
            image_size_mb = base_image.tell() / (1024 * 1024)

            # If still too large, reduce it further
            while image_size_mb > 8:
                width, height = base_image.size
                base_image = base_image.resize((width // 2, height // 2), Image.LANCZOS)
                image_size_mb = base_image.tell() / (1024 * 1024)
                print(f"Still too large, further resized to {width // 2}x{height // 2}. Current size: {image_size_mb:.2f}MB")

        final_image = base_image.copy()

        # Overlay the image on itself 10 times
        for _ in range(10):
            final_image = Image.alpha_composite(final_image, base_image)

        # Save the base and final overlaid images
        if not self.imagePath.lower().endswith('.png'):
            base_image_path = os.path.join(self.image_folder_path, f"base_image_{self.sampleID}.png")
            base_image.save(base_image_path)
            print(f"Base image saved as: {base_image_path}")

        # Save the final overlaid image with a new name
        final_image_name = "final_" + self.sampleID + ".png"
        self.imagePath = os.path.join(self.image_folder_path, final_image_name)
        final_image.save(self.imagePath)
        print(f"Final overlaid image saved as: {self.imagePath}")

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
        print(f"Final overlaid image saved as: {final_image_path}")


    def processImageWithMeshing(self):
        """
        Divides the image into 16 equally sized blocks (4x4 grid)
        and ensures each block is under 8MB.
        Passes each block to the `pureOverlayImage` method for further processing.
        Inputs: None
        Outputs: None
        """

        if not os.path.exists(self.evenLightingImagePath):
            print(f"Error: Image {self.evenLightingImagePath} not found at {self.evenLightingImagePath}")
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
                    scale_factor = (8 / block_size_mb) ** 0.5  # Calculate scaling factor
                    new_width = int(block.width * scale_factor)
                    new_height = int(block.height * scale_factor)
                    block = block.resize((new_width, new_height), Image.LANCZOS)

                    # Recalculate block size after resizing
                    with io.BytesIO() as temp_buffer:
                        block.save(temp_buffer, format="PNG")
                        block_size_mb = temp_buffer.tell() / (1024 * 1024)

                # Append the resized block to the list
                blocks.append(block)

        print(f"Image divided into {len(blocks)} blocks, and each block is under 8MB.")

        # Process each block with `pureOverlayImage`
        for i  in range(1,len(blocks)+1):
            print(type(blocks[i-1]))
            self.pureOverlayImage(blocks[i-1],i)



    def even_out_lighting(self):
        """
        Even out the lighting in the image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to improve the contrast and smooth out lighting inconsistencies.

        Inputs:None

        Outputs:None
        """
        # Load the image
        image = cv2.imread(self.imagePath, cv2.IMREAD_COLOR)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)# Convert to LAB color space to separate intensity from color information
        
        l, a, b = cv2.split(lab_image)# Split the LAB image into its channels
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))  # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Perform a light normalization to smooth out lighting inconsistencies without over-smoothing
        enhanced_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply a slight Gaussian blur to avoid too much noise while keeping details
        final_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
        
        self.imagePath=os.path.join(self.image_folder_path,f"even_lighting_{self.imageName}")
        ## self.evenLightingImagePath
        self.evenLightingImagePath=self.imagePath
        cv2.imwrite(self.imagePath, final_image)
        print(f"Evened out lighting picture saved as : {self.imagePath}")

    def even_out_lighting_validation(self,parameter_folder_path):
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
        enhanced_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)

        # Apply a slight Gaussian blur to avoid too much noise while keeping details
        final_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
        original_folder_path =  self.image_folder_path
        self.image_folder_path=os.path.join(original_folder_path, parameter_folder_path)
        os.makedirs(self.image_folder_path, exist_ok=True)
        self.imagePath = os.path.join(self.image_folder_path, f"even_lighting_{self.imageName}")
        ## self.evenLightingImagePath
        self.evenLightingImagePath = self.imagePath
        cv2.imwrite(self.imagePath, final_image)
        print(f"Evened out lighting picture saved as : {self.imagePath}")

    def cropImage(self):
        """
        Allows the user to manually select a region of interest (ROI) and crop the image to that region.
        
        Inputs:None

        Outputs:None
        """
        roi_selector = ROI.ROISelector(self.imagePath)
        cropped_image = roi_selector.select_and_move_roi()
        self.imagePath = os.path.join(self.image_folder_path, f"cropped_image_{self.imageName}")
        # Save the cropped image to the image folder
        cv2.imwrite(self.imagePath, cropped_image)
        print(f"Cropped image picture saved as : {self.imagePath}")
    def __get_rgb_from_temperature(self,temp):
        """
        Calculate the RGB values of the white point based on color temperature (Kelvin)
        """
        temp = max(1000, min(temp, 40000)) / 100.0

        # Calculate the Red component
        if temp <= 66:
            r = 255
        else:
            tmpCalc = temp - 55
            r = 351.976905668057 + 0.114206453784165 * tmpCalc - 40.2536630933213 * np.log(tmpCalc)
            r = min(255, max(0, r))

        # Calculate the Green component
        if temp <= 66:
            tmpCalc = temp - 2
            g = -155.254855627092 - 0.445969504695791 * tmpCalc + 104.492161993939 * np.log(tmpCalc)
            g = min(255, max(0, g))
        else:
            tmpCalc = temp - 50
            g = 325.449412571197 + 0.0794345653666234 * tmpCalc - 28.0852963507957 * np.log(tmpCalc)
            g = min(255, max(0, g))

        # Calculate the Blue component
        if temp >= 66:
            b = 255
        else:
            if temp <= 19:
                b = 0
            else:
                tmpCalc = temp - 10
                b = -254.769351841209 + 0.827409606400739 * tmpCalc + 115.679944010661 * np.log(tmpCalc)
                b = min(255, max(0, b))

        return np.array([r, g, b], dtype=np.uint8).reshape(1, 1, 3)

    def __color_error(self,rgb1, rgb2):
        diff = np.array(rgb1) - np.array(rgb2)
        return math.sqrt(np.sum(diff ** 2))
    def __get_temperature_from_rgb(self,target_r, target_g, target_b):
        target_rgb = (target_r, target_g, target_b)
        start_time = time.time()
        min_error = float('inf')
        best_temp = 1000

        # Full range 1K step-by-step search
        for temp in range(1000, 40001, 1):
            rgb = self.__get_rgb_from_temperature(temp)
            err = self.__color_error(rgb, target_rgb)
            if err < min_error:
                min_error = err
                best_temp = temp

        end_time = time.time()
        print("Searching consumption time: {:.2f}s".format(end_time - start_time))
        return best_temp, min_error

    def __estimate_temperature_from_image(self):
        image = Image.open(self.imagePath).convert("RGB")
        arr = np.array(image)
        avg_rgb = np.mean(arr.reshape(-1, 3), axis=0)
        avg_r, avg_g, avg_b = avg_rgb

        # estimated_temp, error = get_temperature_from_rgb(avg_r, avg_g, avg_b)
        estimated_temp, error =self.__get_temperature_from_rgb(avg_r, avg_g, avg_b)
        return estimated_temp, error

    def __kelvin_to_rgb(self,temp_kelvin):

        if temp_kelvin < 1000:
            temp_kelvin = 1000
        elif temp_kelvin > 40000:
            temp_kelvin = 40000

        tmp_internal = temp_kelvin / 100.0

        # red
        if tmp_internal <= 66:
            red = 255
        else:
            tmp_red = 329.698727446 * math.pow(tmp_internal - 60, -0.1332047592)
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
            tmp_green = 288.1221695283 * math.pow(tmp_internal - 60, -0.0755148492)
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
            tmp_blue = 138.5177312231 * math.log(tmp_internal - 10) - 305.0447927307
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
        print(f"from temp {from_temp}")
        print(f"To temp {to_temp}")
        adjusted = (image * balance).clip(0, 255).astype(np.uint8)

        return adjusted

    def __color_correction(self,imageTemp,adjustedColorTemp):
        image = cv2.imread(self.imagePath)

        result_image = self.__adjust_temperature(image, imageTemp,adjustedColorTemp)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # save the result image
        self.colorCorrection_imagePath = os.path.join(os.path.dirname(self.imagePath),
                                                      f"{self.sampleID}_color_correction.png")
        print(f"Saving color corrected image to {self.colorCorrection_imagePath}")

        self.imagePath=self.colorCorrection_imagePath
        # Save the result image
        if not cv2.imwrite(self.colorCorrection_imagePath, result_image):
            print(f"Failed to save the color corrected image. Check file path and permissions.")

    ### for image color correction--hardcoded for now
    def colorCorrection(self,adjustedColorTemp):

        image_temp, err = self.__estimate_temperature_from_image()

        if image_temp is None:
            logger.error("Can not get the original color temperature of the sample ID: {}",self.sampleID)
            return
        if self.sampleID.startswith('RCB1489190'):
            image_temp=2648
        if self.sampleID.startswith('RCB1751016'):
            image_temp = 2561
        if self.sampleID.startswith('RCB1763362'):
            image_temp = 2500
        if self.sampleID.startswith('RCB1763004'):
            image_temp = 2444
        if self.sampleID.startswith('RCB1763013'):
            image_temp = 2537
        if self.sampleID.startswith('RCB1754033'):
            image_temp = 2513
        if self.sampleID.startswith('RCB1766399'):
            image_temp = 2513
        if self.sampleID.startswith('RCB1767022'):
            image_temp = 2513

        print(f"The original color temperature of the sample ID: {self.sampleID} is {image_temp}")
        print(f"Target color temperature is {adjustedColorTemp}")
        self.__color_correction(image_temp,adjustedColorTemp)

